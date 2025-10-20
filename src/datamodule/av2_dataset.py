from typing import List
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm
import re

_FRAME_ID_RE = re.compile(r"log_([^_]+)_(\d+)")

def get_frame_id(file_path: Path) -> int:
    """파일 경로에서 프레임 ID를 정수형으로 추출합니다."""
    match = _FRAME_ID_RE.search(file_path.stem)
    if match:
        return int(match.group(2))
    return -1

class Av2Dataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str = None,
        num_historical_steps: int = 20,
        sequence_origins: List[int] = [20],
        radius: float = 150.0,
        train_mode: str = 'only_focal',

    ):
        assert sequence_origins[-1] == 20 and num_historical_steps <= 20
        assert train_mode in ['only_focal', 'focal_and_scored']
        assert split in ['train', 'val', 'test']
        super(Av2Dataset, self).__init__()

        self.data_folder = Path(data_root) / split
        # 파일을 시나리오와 프레임 순서로 정렬합니다.
        self.file_list = sorted(list(self.data_folder.glob('*.pt')), key=lambda p: (p.stem.split('_')[1], get_frame_id(p)))
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = 0 if split =='test' else 60
        self.sequence_origins = sequence_origins
        self.mode = 'only_focal' if split != 'train' else train_mode
        self.radius = radius

        print(f'data root: {data_root}/{split}, total number of files: {len(self.file_list)}')

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int):
        data = torch.load(self.file_list[index])
        data = self.process(data)
        return data
    
    def process(self, data):
        sequence_data = []
        train_idx = [data['focal_idx']]
        
        # 'only_focal' for single-agent setting, 'focal_and_scored' for multi-agent setting
        train_idx += data['scored_idx']
        
        for cur_step in self.sequence_origins:
            for ag_idx in train_idx:
                ag_dict = self.process_single_agent(data, ag_idx, cur_step)
                sequence_data.append(ag_dict)
        
        return sequence_data

    def process_single_agent(self, data, idx, step=20):
        # info for cur_agent on cur_step
        cur_agent_id = data['agent_ids'][idx]
        origin = data['x_positions'][idx, step - 1].double()
        theta = data['x_angles'][idx, step - 1].double()
        rotate_mat = torch.tensor(
            [
                [torch.cos(theta), -torch.sin(theta)],
                [torch.sin(theta), torch.cos(theta)],
            ],
        )
        ag_mask = torch.norm(data['x_positions'][:, step - 1] - origin, dim=-1) < self.radius
        ag_mask = ag_mask * data['x_valid_mask'][:, step - 1]
        ag_mask[idx] = False

        # transform agents to local
        st, ed = step - self.num_historical_steps, step + self.num_future_steps
        attr = torch.cat([data['x_attr'][[idx]], data['x_attr'][ag_mask]])
        pos = data['x_positions'][:, st: ed]
        pos = torch.cat([pos[[idx]], pos[ag_mask]])
        head = data['x_angles'][:, st: ed]
        head = torch.cat([head[[idx]], head[ag_mask]])
        vel = data['x_velocity'][:, st: ed]
        vel = torch.cat([vel[[idx]], vel[ag_mask]])
        valid_mask = data['x_valid_mask'][:, st: ed]
        valid_mask = torch.cat([valid_mask[[idx]], valid_mask[ag_mask]])

        pos[valid_mask] = torch.matmul(pos[valid_mask].double() - origin, rotate_mat).to(torch.float32)
        head[valid_mask] = (head[valid_mask] - theta + np.pi) % (2 * np.pi) - np.pi

        # transform lanes to local
        l_pos = data['lane_positions']
        l_attr = data['lane_attr']
        l_is_int = data['is_intersections']
        l_pos = torch.matmul(l_pos.reshape(-1, 2).double() - origin, rotate_mat).reshape(-1, l_pos.size(1), 2).to(torch.float32)


        l_ctr = l_pos[:, 9:11].mean(dim=1)
        l_head = torch.atan2(
            l_pos[:, 10, 1] - l_pos[:, 9, 1],
            l_pos[:, 10, 0] - l_pos[:, 9, 0],
        )
        l_valid_mask = (
            (l_pos[:, :, 0] > -self.radius) & (l_pos[:, :, 0] < self.radius)
            & (l_pos[:, :, 1] > -self.radius) & (l_pos[:, :, 1] < self.radius)
        )

        l_mask = l_valid_mask.any(dim=-1)
        l_pos = l_pos[l_mask]
        l_is_int = l_is_int[l_mask]
        l_attr = l_attr[l_mask]
        l_ctr = l_ctr[l_mask]
        l_head = l_head[l_mask]
        l_valid_mask = l_valid_mask[l_mask]

        l_pos = torch.where(
            l_valid_mask[..., None], l_pos, torch.zeros_like(l_pos)
        )

        # remove outliers
        nearest_dist = torch.cdist(pos[:, self.num_historical_steps - 1, :2],
                                   l_pos.view(-1, 2)).min(dim=1).values
        ag_mask = nearest_dist < 5
        ag_mask[0] = True
        pos = pos[ag_mask]
        head = head[ag_mask]
        vel = vel[ag_mask]
        attr = attr[ag_mask]
        valid_mask = valid_mask[ag_mask]
        # ===== lane-aware: find nearest lane for focal agent ===== #add eun
        focal_cur_pos = pos[0, self.num_historical_steps - 1, :2].float()

        if l_pos.size(0) > 0:
            # 1) 차선 중심과의 거리로 가장 가까운 lane 선택
            lane_centers = l_pos[:, 9:11].mean(dim=1)
            lane_dists_to_center = torch.norm(lane_centers - focal_cur_pos[None, :], dim=-1)
            nearest_lane_idx = torch.argmin(lane_dists_to_center)
            nearest_lane_form = l_attr[nearest_lane_idx, 1].long().view(1)

            # 2) focal 위치에서 각 차선 포인트까지 최소 거리 (offset)
            flat_l_pos = l_pos.reshape(-1, 2)
            flat_l_valid = l_valid_mask.reshape(-1)
            valid_flat_l_pos = flat_l_pos[flat_l_valid].float()

            min_offset_dist = torch.tensor([[8.0]]).float()  # 기본값 [1,1]

            if valid_flat_l_pos.size(0) > 0:
                dists_to_all_lane_points = torch.cdist(focal_cur_pos.unsqueeze(0), valid_flat_l_pos)
                min_offset_dist = dists_to_all_lane_points.min().float().unsqueeze(0).unsqueeze(1)

            if min_offset_dist.item() > 8.0:
                nearest_lane_form = torch.tensor([0]).long().view(1)
                min_offset_dist = torch.tensor([[8.0]]).float()
        else:
            nearest_lane_form = torch.tensor([0]).long().view(1)
            min_offset_dist = torch.tensor([[8.0]]).float()
        # ============================================================

        # post_process
        head = head[:, :self.num_historical_steps]
        vel_future = vel[:, self.num_historical_steps:]
        vel = vel[:, :self.num_historical_steps]
        pos_ctr = pos[:, self.num_historical_steps - 1].clone()
        if self.num_future_steps > 0:
            type_mask = attr[:, [-1]] != 3
            pos, target = pos[:, :self.num_historical_steps], pos[:, self.num_historical_steps:]
            target_mask = type_mask & valid_mask[:, [self.num_historical_steps - 1]] & valid_mask[:, self.num_historical_steps:]
            valid_mask = valid_mask[:, :self.num_historical_steps]
            target = torch.where(
                target_mask.unsqueeze(-1),
                target - pos_ctr.unsqueeze(1), torch.zeros(pos_ctr.size(0), 60, 2),   
            )
        else:
            target = target_mask = None

        diff_mask = valid_mask[:, :self.num_historical_steps - 1] & valid_mask[:, 1: self.num_historical_steps]
        tmp_pos = pos.clone()
        pos_diff = pos[:, 1:self.num_historical_steps] - pos[:, :self.num_historical_steps - 1]
        
        # add target velocity and acceleration 
        target_diff = None
        if target is not None:
            target_diff_tmp = torch.cat((pos[:, -1].unsqueeze(1), target), dim=1)
            target_diff = target_diff_tmp[:, 1:self.num_future_steps+1] - target_diff_tmp[:, :self.num_future_steps]
            target_diff_tmp = target_diff.clone()
            diff_mask_target_tmp = torch.cat((valid_mask[:,-1].unsqueeze(1), target_mask), dim=1)
            diff_mask_target = diff_mask_target_tmp[:, 1:self.num_future_steps + 1] & diff_mask_target_tmp[:, : self.num_future_steps]
            target_diff[:, :] = torch.where(
                diff_mask_target.unsqueeze(-1),
                target_diff_tmp, torch.zeros(target_diff_tmp.size(0), self.num_future_steps, 2)
            )
        
        pos[:, 1:self.num_historical_steps] = torch.where(
            diff_mask.unsqueeze(-1),
            pos_diff, torch.zeros(pos.size(0), self.num_historical_steps - 1, 2)
        )
        pos[:, 0] = torch.zeros(pos.size(0), 2)

        tmp_vel = vel.clone()
        vel_diff = vel[:, 1:self.num_historical_steps] - vel[:, :self.num_historical_steps - 1]
        vel[:, 1:self.num_historical_steps] = torch.where(
            diff_mask,
            vel_diff, torch.zeros(vel.size(0), self.num_historical_steps - 1)
        )
        vel[:, 0] = torch.zeros(vel.size(0))
        
        # add target velocity and acceleration
        if target is not None:
            tmpvel_future = vel_future.clone()
            tmpvel_future = torch.cat((tmp_vel[:, -1].unsqueeze(1), tmpvel_future), dim=1)
            vel_diff_future= tmpvel_future[:, 1:self.num_future_steps+1] - tmpvel_future[:, :self.num_future_steps]
            vel_future[:, :] = torch.where(
                diff_mask_target,
                vel_diff_future, torch.zeros(vel_diff_future.size(0), self.num_future_steps)
            )
        
        return {
            'target': target,
            'target_diff': target_diff,
            'target_vel_diff': vel_future,
            'target_mask': target_mask,

            'x_positions_diff': pos,
            'x_positions': tmp_pos,
            'x_attr': attr,
            'x_centers': pos_ctr,
            'x_angles': head,
            'x_velocity': tmp_vel,
            'x_velocity_diff': vel,
            'x_valid_mask': valid_mask,

            'lane_positions': l_pos,
            'lane_centers': l_ctr,
            'lane_angles': l_head,
            'lane_attr': l_attr,
        
            'lane_valid_mask': l_valid_mask,
            'is_intersections': l_is_int,
            
            'origin': origin.view(1, 2),
            'theta': theta.view(1),
            'scenario_id': data['scenario_id'],
            'track_id': cur_agent_id,
            'city': data['city'],
            'timestamp': torch.Tensor([step * 0.1]),
            'lane_form_type': nearest_lane_form,  # add eun
            'lane_offset_dist': min_offset_dist,
        }

def collate_fn(seq_batch):
    # seq_batch는 [[scene1_agents], [scene2_agents], ...] 형태입니다.
    # 이것을 [agent1, agent2, agent3, ...] 형태의 단일 리스트로 만듭니다.
    flat_batch = []
    for sample in seq_batch:
        for agent in sample:
            flat_batch.append(agent)

    if not flat_batch:
        return {}

    data = {}
    
    # 이제부터는 `flat_batch`를 사용하여 모든 작업을 수행합니다.
    for key in [
        'x_positions_diff', 'x_attr', 'x_positions', 'x_centers',
        'x_angles', 'x_velocity', 'x_velocity_diff',
        'lane_positions', 'lane_centers', 'lane_angles', 'lane_attr',
        'is_intersections'
    ]:
        if key in flat_batch[0]:
            data[key] = pad_sequence([b[key] for b in flat_batch], batch_first=True)

    if 'x_scored' in flat_batch[0]:
        data['x_scored'] = pad_sequence([b['x_scored'] for b in flat_batch], batch_first=True)

    if all(b.get('target') is not None for b in flat_batch):
        data['target']         = pad_sequence([b['target'] for b in flat_batch], batch_first=True)
        data['target_diff']    = pad_sequence([b['target_diff'] for b in flat_batch], batch_first=True)
        data['target_vel_diff']= pad_sequence([b['target_vel_diff'] for b in flat_batch], batch_first=True)
        data['target_mask']    = pad_sequence([b['target_mask'] for b in flat_batch], batch_first=True, padding_value=False)

    for key in ['x_valid_mask', 'lane_valid_mask']:
        data[key] = pad_sequence([b[key] for b in flat_batch], batch_first=True, padding_value=False)

    data['x_key_valid_mask']   = data['x_valid_mask'].any(-1)
    data['lane_key_valid_mask']= data['lane_valid_mask'].any(-1)

    data['scenario_id'] = [b['scenario_id'] for b in flat_batch]
    data['track_id']    = [b['track_id'] for b in flat_batch]

    data['origin']   = torch.cat([b['origin']   for b in flat_batch], dim=0)
    data['theta']    = torch.cat([b['theta']    for b in flat_batch], dim=0)
    data['timestamp']= torch.cat([b['timestamp']for b in flat_batch], dim=0)
    
    if 'lane_form_type' in flat_batch[0]:
        data['lane_form_type'] = torch.cat([b['lane_form_type'] for b in flat_batch])
        
    if 'lane_offset_dist' in flat_batch[0]:
        data['lane_offset_dist'] = torch.cat([b['lane_offset_dist'] for b in flat_batch], dim=0)

    return data