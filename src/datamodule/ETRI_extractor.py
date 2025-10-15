import pickle
import traceback
from pathlib import Path
from typing import List

import numpy as np
import torch


def _resample_polyline_xy(pts: np.ndarray, num_points: int = 20) -> torch.Tensor:
    """
    pts: (L, 2) numpy array
    return: (num_points, 2) torch.float32
    """
    if pts.ndim != 2 or pts.shape[-1] != 2:
        pts = np.asarray(pts).reshape(-1, 2)
    L = len(pts)
    if L == 0:
        return torch.zeros(num_points, 2, dtype=torch.float32)
    if L == 1:
        return torch.tensor(np.repeat(pts, num_points, axis=0), dtype=torch.float32)

    # 누적 호 길이
    seg = np.diff(pts, axis=0)
    dist = np.sqrt((seg ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(dist)])
    total = s[-1]
    if total <= 1e-6:
        # 모든 점이 동일
        return torch.tensor(np.repeat(pts[:1], num_points, axis=0), dtype=torch.float32)

    # 균등 간격 목표 길이
    s_target = np.linspace(0.0, total, num_points)

    # 각 좌표 축을 s에 대해 보간
    x = np.interp(s_target, s, pts[:, 0])
    y = np.interp(s_target, s, pts[:, 1])
    out = np.stack([x, y], axis=-1).astype(np.float32)
    return torch.from_numpy(out)


class ETRIExtractor:
    def __init__(
        self,
        radius: float = 150.0,
        save_path: Path | None = None,
        mode: str = "train",
        remove_outlier_actors: bool = True,
        lane_num_interp_pts: int = 20,  # AV2 extractor와 동일하게 lane 길이 고정
    ) -> None:
        self.save_path = save_path
        self.mode = mode
        self.radius = radius
        self.remove_outlier_actors = remove_outlier_actors
        self.lane_num_interp_pts = lane_num_interp_pts

    def save(self, file: Path):
        assert self.save_path is not None
        try:
            data = self.get_data(file)
        except Exception:
            print(traceback.format_exc())
            print(f"found error while extracting data from {file}")
            return
        save_file = self.save_path / (file.stem + ".pt")
        torch.save(data, save_file)

    def get_data(self, file: Path):
        return self.process(file)

    def process(self, raw_path: Path):
        """
        ETRI raw .pkl -> DeMo(AV2) 호환 dict
        기대하는 파일 구조 예:
            sample = {
                "log_id": str,
                "agent": {
                    "id": List[str],
                    "type": np.ndarray[int] or list[int],        # (N,)
                    "category": np.ndarray[int] or list[int],    # (N,)
                    "position": np.ndarray[float],               # (N, T, 2)
                    "heading": np.ndarray[float],                # (N, T)
                    "velocity": ... (사용 안함, 전부 -1000)
                    "valid_mask": np.ndarray[bool],              # (N, T)
                },
                "map": List[{
                    "ID": ...,
                    "Type": int,
                    "LLinkID": int,
                    "RLinkID": int,
                    "Pts": np.ndarray (L, 2 or >=2),
                    "Speed": float (optional),
                    ...
                }]
            }
        """
        with open(raw_path, "rb") as f:
            sample = pickle.load(f)

        agents = sample["agent"]
        lane_segments = sample["map"]
        scenario_id = raw_path.stem.replace("log_", "", 1)

        # ---------- Agents ----------
        pos_np = np.asarray(agents["position"], dtype=np.float32)[..., :2]  # (N, T, 2), meter 단위
        x_positions = torch.from_numpy(pos_np)

        # heading (N, T)
        x_angles = torch.tensor(agents["heading"], dtype=torch.float32)

        # velocity: Δpos / Δt (Δt=0.1s → 10Hz)
        diff = np.diff(pos_np, axis=1)  # (N, T-1, 2)
        speed = np.linalg.norm(diff, axis=-1) / 0.1  # (N, T-1), m/s
        first_step = speed[:, :1]
        speed = np.concatenate([first_step, speed], axis=1)  # (N, T)
        x_velocity = torch.from_numpy(speed).float()

        # valid mask
        vm = np.asarray(agents["valid_mask"]).astype(bool)  # (N, T)
        x_valid_mask = torch.from_numpy(vm)

        # x_attr: [type, category, combined_type]
        type_np = np.asarray(agents["type"], dtype=np.int64)
        cat_np = np.asarray(agents["category"], dtype=np.int64)
        comb_np = cat_np.copy()
        x_attr = torch.stack(
            [
                torch.from_numpy(type_np),
                torch.from_numpy(cat_np),
                torch.from_numpy(comb_np),
            ],
            dim=-1,
        ).long()  # (N, 3)

        agent_ids = list(map(str, agents["id"]))
        focal_idx = agent_ids.index("-1") if "-1" in agent_ids else 0
        scored_idx = [i for i, c in enumerate(cat_np.tolist()) if c == 2 and i != focal_idx]

        # ---------- Lanes ----------
        lane_dict = {seg["ID"]: seg for seg in lane_segments}

        lane_positions_list: List[torch.Tensor] = []
        is_intersections_list: List[float] = []
        lane_attr_list: List[torch.Tensor] = []

        for seg in lane_segments:
            pts_np = np.asarray(seg["Pts"], dtype=np.float32)
            pts_xy = pts_np[..., :2]
            lane_centerline = _resample_polyline_xy(pts_xy, num_points=self.lane_num_interp_pts)
            lane_positions_list.append(lane_centerline)

            lane_type = int(seg.get("Type", 0))
            is_intersection = float(lane_type in [2, 3, 4, 5])
            is_intersections_list.append(is_intersection)
            lane_type = 0  # 단순화

            # --- lane width 계산 ---
            neighbor_widths = []
            for neighbor_key in ["LLinkID", "RLinkID"]:
                neighbor_id = seg.get(neighbor_key, -1)
                if neighbor_id != -1 and neighbor_id in lane_dict:
                    neighbor_pts = np.asarray(lane_dict[neighbor_id]["Pts"], dtype=np.float32)[..., :2]
                    neighbor_resampled = _resample_polyline_xy(neighbor_pts, num_points=self.lane_num_interp_pts)

                    widths = torch.norm(lane_centerline - neighbor_resampled, dim=-1)  # (20,)
                    neighbor_widths.append(widths.mean().item())

            if len(neighbor_widths) > 0:
                lane_width = float(np.mean(neighbor_widths))
            else:
                lane_width = 3.5  # 기본 폭 (m)

            lane_attr_list.append(
                torch.tensor([float(lane_type), lane_width, is_intersection], dtype=torch.float32)
            )

        if len(lane_positions_list) == 0:
            lane_positions = torch.zeros(1, self.lane_num_interp_pts, 2, dtype=torch.float32)
            is_intersections = torch.zeros(1, dtype=torch.float32)
            lane_attr = torch.zeros(1, 3, dtype=torch.float32)
        else:
            lane_positions = torch.stack(lane_positions_list, dim=0)
            is_intersections = torch.tensor(is_intersections_list, dtype=torch.float32)
            lane_attr = torch.stack(lane_attr_list, dim=0)

        # ---------- Return ----------
        return {
            "x_positions": x_positions,           # (N, T, 2)
            "x_attr": x_attr,                     # (N, 3)
            "x_angles": x_angles,                 # (N, T)
            "x_velocity": x_velocity,             # (N, T), m/s
            "x_valid_mask": x_valid_mask,         # (N, T)
            "lane_positions": lane_positions,     # (M, 20, 2)
            "lane_attr": lane_attr,               # (M, 3) [lane_type, lane_width, is_intersection]
            "is_intersections": is_intersections, # (M,)
            "scenario_id": scenario_id,           # str
            "agent_ids": agent_ids,               # list[str]
            "focal_idx": focal_idx,               # int
            "scored_idx": scored_idx,             # list[int]
            "city": "ETRI",                       # placeholder
        }
