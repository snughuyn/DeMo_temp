import datetime
from pathlib import Path
import time
import pickle
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.metrics import MR, minADE, minFDE, brier_minFDE
from src.utils.optim import WarmupCosLR
from src.utils.submission_av2 import SubmissionAv2
from src.utils.LaplaceNLLLoss import LaplaceNLLLoss
from .model_forecast import ModelForecast, StreamModelForecast
from src.utils import nll_loss_gmm_direct
from src.utils import compute_temporal_loss 
from collections import defaultdict

class Trainer(pl.LightningModule):
    def __init__(
        self,
        model: dict,
        pretrained_weights: str = None,
        lr: float = 1e-3,
        warmup_epochs: int = 10,
        epochs: int = 60,
        weight_decay: float = 1e-4,
    ) -> None:
        super(Trainer, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        self.submission_handler = SubmissionAv2()

        model_type = model.pop('type')

        self.net = self.get_model(model_type)(**model)

        if pretrained_weights is not None:
            self.net.load_from_checkpoint(pretrained_weights)
            print('Pretrained weights have been loaded.')

        metrics = MetricCollection(
            {
                "minADE1": minADE(k=1),
                "minADE6": minADE(k=6),
                "minFDE1": minFDE(k=1),
                "minFDE6": minFDE(k=6),
                "MR": MR(),
                "b-minFDE6": brier_minFDE(k=6),
            }
        )
        self.laplace_loss = LaplaceNLLLoss()
        self.val_metrics = metrics.clone(prefix="val_")
        self.val_metrics_new = metrics.clone(prefix="val_new_")
    
    def get_model(self, model_type):
        model_dict = {
            'ModelForecast': ModelForecast,  # only 'DeMo'
            'StreamModelForecast': StreamModelForecast,  # integrate 'DeMo' with 'RealMotion'
        }
        assert model_type in model_dict
        return model_dict[model_type]

    def forward(self, data):
        return self.net(data)

    def predict(self, data):
        memory_dict = None
        predictions = []
        probs = []
        for i in range(len(data)):
            cur_data = data[i]
            cur_data['memory_dict'] = memory_dict
            out = self(cur_data)
            memory_dict = out['memory_dict']
            prediction, prob = self.submission_handler.format_data(
                cur_data, out["y_hat"], out["pi"], inference=True)
            predictions.append(prediction)
            probs.append(prob)

        return predictions, probs
    
    def get_dense_future_prediction_loss(self, tb_pre_tag='', tb_dict=None, disp_dict=None):
        if not hasattr(self.net.time_decoder, "forward_ret_dict"):
            return torch.tensor(0.0, device=self.device), tb_dict or {}, disp_dict or {}

        forward_ret = self.net.time_decoder.forward_ret_dict
        if "pred_dense_trajs" not in forward_ret:
            return torch.tensor(0.0, device=self.device), tb_dict or {}, disp_dict or {}

        pred_dense_trajs = forward_ret["pred_dense_trajs"]              # (B, N_all, T, 7)
        obj_trajs_future_state = forward_ret["obj_trajs_future_state"]  # (B, N_all, T, 4)
        obj_trajs_future_mask = forward_ret["obj_trajs_future_mask"]    # (B, N_all, T)

        # --- type 정보 가져오기 ---
        if "x_attr" not in forward_ret:
            return torch.tensor(0.0, device=self.device), tb_dict or {}, disp_dict or {}

        x_attr = forward_ret["x_attr"]                                  # (B, N_actor, 3)
        agent_type = x_attr[..., 0]                                     # (B, N_actor)
        mask_non_ego = (agent_type != 0)

        B, N_all, T, _ = pred_dense_trajs.shape
        N_actor = x_attr.shape[1]

        # --- actor 영역만 사용 ---
        pred_dense_trajs = pred_dense_trajs[:, :N_actor]                # (B, N_actor, T, 7)
        obj_trajs_future_state = obj_trajs_future_state[:, :N_actor]    # (B, N_actor, T, 4)
        obj_trajs_future_mask = obj_trajs_future_mask[:, :N_actor]      # (B, N_actor, T)

        # --- 필터링 마스크 (EGO 제외 & 유효 trajectory만) ---
        valid_mask = mask_non_ego & (obj_trajs_future_mask.sum(dim=-1) > 0)  # (B, N_actor)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device), tb_dict or {}, disp_dict or {}

        # === Flatten valid agents ===
        pred_dense_trajs = pred_dense_trajs[valid_mask]
        obj_trajs_future_state = obj_trajs_future_state[valid_mask]
        obj_trajs_future_mask = obj_trajs_future_mask[valid_mask]

        # === 분리 ===
        pred_dense_trajs_gmm = pred_dense_trajs[..., 0:5]   # [V, T, 5]
        pred_dense_trajs_vel = pred_dense_trajs[..., 5:7]   # [V, T, 2]

        # === velocity loss ===
        loss_vel = F.l1_loss(
            pred_dense_trajs_vel,
            obj_trajs_future_state[..., 2:4],
            reduction='none'
        )
        loss_vel = (loss_vel * obj_trajs_future_mask[..., None]).sum(dim=-1).mean()

        # === GMM NLL loss ===
        V = pred_dense_trajs.shape[0]
        fake_scores = pred_dense_trajs.new_zeros((V, 1))
        temp_pred_trajs = pred_dense_trajs_gmm.unsqueeze(1)  # (V, 1, T, 5)
        temp_gt_idx = torch.zeros(V, dtype=torch.long, device=pred_dense_trajs.device)
        temp_gt_trajs = obj_trajs_future_state[..., 0:2]
        temp_gt_trajs_mask = obj_trajs_future_mask

        loss_gmm, _ = nll_loss_gmm_direct(
            pred_scores=fake_scores,
            pred_trajs=temp_pred_trajs,
            gt_trajs=temp_gt_trajs,
            gt_valid_mask=temp_gt_trajs_mask,
            pre_nearest_mode_idxs=temp_gt_idx,
            timestamp_loss_weight=None,
            use_square_gmm=False,
        )
        loss_gmm = loss_gmm.mean()

        # === combine velocity + GMM ===
        loss_reg = loss_gmm + loss_vel

        # === Logging ===
        if tb_dict is None: tb_dict = {}
        if disp_dict is None: disp_dict = {}

        tb_dict[f'{tb_pre_tag}loss_dense_prediction'] = loss_reg.item()
        disp_dict[f'{tb_pre_tag}loss_dense_prediction'] = loss_reg.item()

        return loss_reg, tb_dict, disp_dict


    def get_dense_future_prediction_loss(self, pred_dense_trajs, obj_trajs_future_state, obj_trajs_future_mask, x_attr, tb_pre_tag='', tb_dict=None, disp_dict=None):
        """
        [수정된 함수]
        DFP (Dense Future Prediction) 손실을 계산합니다.
        모든 데이터는 공유 딕셔너리가 아닌, 함수의 인자로 직접 전달받습니다.
        """
        # 1. 인자로 받은 pred_dense_trajs를 직접 확인합니다.
        if pred_dense_trajs is None:
            return torch.tensor(0.0, device=x_attr.device), tb_dict or {}, disp_dict or {}

        # 2. 인자로 받은 데이터들을 사용합니다. (공유 딕셔너리 접근 코드 삭제)
        agent_type = x_attr[..., 0]
        mask_valid_agent = (agent_type != 0)

        # --- 유효 trajectory가 있는 agent만 필터링 ---
        has_future = (obj_trajs_future_mask.sum(dim=-1) > 0)
        valid_mask = mask_valid_agent & has_future

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=x_attr.device), tb_dict or {}, disp_dict or {}

        # --- valid agent만 선택 ---
        pred_dense_trajs = pred_dense_trajs[valid_mask]
        obj_trajs_future_state = obj_trajs_future_state[valid_mask]
        obj_trajs_future_mask = obj_trajs_future_mask[valid_mask]

        # --- 분리 ---
        pred_dense_trajs_gmm = pred_dense_trajs[..., 0:5]
        pred_dense_trajs_vel = pred_dense_trajs[..., 5:7]

        # === velocity loss ===
        loss_vel = F.l1_loss(
            pred_dense_trajs_vel,
            obj_trajs_future_state[..., 2:4],
            reduction='none'
        )
        loss_vel = (loss_vel * obj_trajs_future_mask[..., None]).sum(dim=-1).mean()

        # === GMM NLL loss ===
        V = pred_dense_trajs.shape[0]
        fake_scores = pred_dense_trajs.new_zeros((V, 1))
        temp_pred_trajs = pred_dense_trajs_gmm.unsqueeze(1)
        temp_gt_idx = torch.zeros(V, dtype=torch.long, device=pred_dense_trajs.device)
        temp_gt_trajs = obj_trajs_future_state[..., 0:2]
        temp_gt_trajs_mask = obj_trajs_future_mask

        loss_gmm, _ = nll_loss_gmm_direct(
            pred_scores=fake_scores,
            pred_trajs=temp_pred_trajs,
            gt_trajs=temp_gt_trajs,
            gt_valid_mask=temp_gt_trajs_mask,
            pre_nearest_mode_idxs=temp_gt_idx,
            timestamp_loss_weight=None,
            use_square_gmm=False,
        )
        loss_gmm = loss_gmm.mean()

        # === combine velocity + GMM ===
        loss_reg = loss_gmm + loss_vel

        # === Logging ===
        if tb_dict is None: tb_dict = {}
        if disp_dict is None: disp_dict = {}
        tb_dict[f'{tb_pre_tag}loss_dense_prediction'] = loss_reg.item()
        disp_dict[f'{tb_pre_tag}loss_dense_prediction'] = loss_reg.item()

        return loss_reg, tb_dict, disp_dict

    def cal_temporal_loss(self, out, data):
        # Loss 함수 내에서 엄격하게 (i, i+1) 쌍만 비교하도록 로직이 구현되어 있습니다.
        temporal_loss = compute_temporal_loss(
            y_hat=out["y_hat"],
            scenario_ids=data["scenario_id"],
            track_ids=data["track_id"], 
            time_shift=1, 
            bidirectional=True, 
            reduction="mean",
        )
        return temporal_loss
    
    def cal_loss(self, out, data, tag=''):
        """
        [수정된 함수]
        전체 손실을 계산합니다.
        공유 딕셔너리 대신, out 딕셔너리에서 예측값을 직접 가져와 사용합니다.
        """
        # ===== GT 생성 =====
        with torch.no_grad():
            gt_pos = data["target"]
            gt_mask = data["target_mask"].bool()
            dt = 0.1
            vel = gt_pos.diff(dim=2, prepend=gt_pos[:, :, :1, :]) / dt
            obj_trajs_future_state = torch.cat([gt_pos, vel], dim=-1)
            obj_trajs_future_mask = gt_mask

        # ===== type==0 (미관측 agent) 제외용 마스크 =====
        x_attr = data["x_attr"]
        agent_type = x_attr[..., 0]
        mask_valid_agent = (agent_type != 0)

        # ===== 예측 결과 =====
        y_hat = out["y_hat"]
        pi = out["pi"]
        y_hat_others = out["y_hat_others"]
        scal, scal_new = out["scal"], out["scal_new"]
        new_y_hat = out.get("new_y_hat", None)
        new_pi = out.get("new_pi", None)
        dense_predict = out.get("dense_predict", None)

        # ===== GT =====
        y = data["target"][:, 0]
        y_others = data["target"][:, 1:]

        if y_hat_others is not None:
            valid_mask = mask_valid_agent[:, 1:]
            y_hat_others = y_hat_others[valid_mask]
            y_others = y_others[valid_mask]
            others_reg_mask = data["target_mask"][:, 1:][valid_mask]
        else:
            others_reg_mask = None
        
        pred_dense_trajs = out.get("pred_dense_trajs", None)
        
        dense_reg_loss = torch.tensor(0.0, device=y.device)
        if dense_predict is not None:
            tgt_mask = data["target_mask"][:, 0].unsqueeze(-1).float()
            dense_reg = F.smooth_l1_loss(dense_predict, y[..., :2], reduction="none")
            denom = tgt_mask.sum().clamp_min(1.0)
            dense_reg_loss = (dense_reg * tgt_mask).sum() / denom
            dense_reg_loss = torch.nan_to_num(dense_reg_loss, nan=0.0)

        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]
        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
        agent_cls_loss = F.cross_entropy(pi, best_mode.detach(), label_smoothing=0.2)

        if new_y_hat is not None:
            l2_norm_new = torch.norm(new_y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
            best_mode_new = torch.argmin(l2_norm_new, dim=-1)
            new_y_hat_best = new_y_hat[torch.arange(new_y_hat.shape[0]), best_mode_new]
            new_agent_reg_loss = F.smooth_l1_loss(new_y_hat_best[..., :2], y)
            new_pi_reg_loss = F.cross_entropy(new_pi, best_mode_new.detach(), label_smoothing=0.2)
        else:
            new_agent_reg_loss = torch.tensor(0.0, device=y.device)
            new_pi_reg_loss = torch.tensor(0.0, device=y.device)

        if others_reg_mask is not None and others_reg_mask.sum() > 0:
            others_reg_loss = F.smooth_l1_loss(y_hat_others[others_reg_mask], y_others[others_reg_mask])
        else:
            others_reg_loss = torch.tensor(0.0, device=y.device)

        predictions = {"traj": y_hat, "scale": scal, "probs": pi}
        laplace_loss = self.laplace_loss.compute(predictions, y)
        predictions = {"traj": new_y_hat, "scale": scal_new, "probs": new_pi}
        laplace_loss_new = self.laplace_loss.compute(predictions, y)

        loss = (
            agent_reg_loss + agent_cls_loss +
            others_reg_loss +
            new_agent_reg_loss + new_pi_reg_loss +
            laplace_loss + laplace_loss_new + dense_reg_loss
        )

        # ===== DFP Loss 추가 =====
        # 1. out 딕셔너리에서 pred_dense_trajs를 직접 가져옵니다.
        pred_dense_trajs = out.get("pred_dense_trajs", None)
        
        # 2. 함수에 필요한 모든 정보를 인자로 명시적으로 전달합니다.
        dfp_loss, _, _ = self.get_dense_future_prediction_loss(
            pred_dense_trajs=pred_dense_trajs,
            obj_trajs_future_state=obj_trajs_future_state,
            obj_trajs_future_mask=obj_trajs_future_mask,
            x_attr=data["x_attr"], # 현재 배치의 x_attr 사용
        )
        loss = loss + 1e-4 * dfp_loss

        # ===== Logging =====
        disp_dict = {
            f"{tag}loss": loss.item(),
            f"{tag}reg_loss": agent_reg_loss.item(),
            f"{tag}cls_loss": agent_cls_loss.item(),
            f"{tag}others_reg_loss": others_reg_loss.item(),
            f"{tag}laplace_loss": laplace_loss.item(),
            f"{tag}laplace_loss_new": laplace_loss_new.item(),
            f"{tag}dfp_loss": 1e-4 * dfp_loss.item(),
        }
        return loss, disp_dict

    def training_step(self, batch, batch_idx):
        """
        [최종 수정 버전]
        뒤섞인 temporal_batch를 scenario_id와 track_id를 기준으로
        완벽하게 (t, t+1) 쌍으로 재구성하여 손실을 계산합니다.
        """
        main_batch, temporal_batch = batch['main'], batch['temporal']

        # --- 1. Main Loss 계산 (전체 데이터 대상) ---
        out_main = self(main_batch)
        main_loss, loss_dict = self.cal_loss(out_main, main_batch)

        # --- 2. Temporal Loss 계산 (쌍 데이터 재구성) ---
        temporal_loss = torch.tensor(0.0, device=self.device)
        
        # 2-1. temporal_batch 내의 모든 에이전트를 (scenario_id, track_id) 키로 묶습니다.
        agents_map = defaultdict(dict)
        for i in range(len(temporal_batch['scenario_id'])):
            sc_id = temporal_batch['scenario_id'][i]
            tk_id = temporal_batch['track_id'][i]
            
            # scenario_id에서 't' 시점인지 't+1' 시점인지 구분
            try:
                base_id, frame_str = sc_id.rsplit('_', 1)
                frame = int(frame_str)
                # (base_id, tk_id)를 고유 키로 사용
                unique_key = (base_id, tk_id)
                agents_map[unique_key][frame] = i # {프레임: 인덱스} 형태로 저장
            except (ValueError, IndexError):
                continue

        # 2-2. (t, t+1) 쌍이 모두 존재하는 에이전트들의 인덱스를 추출합니다.
        t_indices = []
        t1_indices = []
        for unique_key, frame_map in agents_map.items():
            # t와 t+1 프레임이 모두 있는지 확인
            # PairedSampler가 frame_t, frame_t+1 순서로 데이터를 제공하므로,
            # min, max를 사용하여 t와 t+1을 구분할 수 있습니다.
            if len(frame_map) == 2:
                frame_t = min(frame_map.keys())
                frame_t1 = max(frame_map.keys())
                if frame_t1 == frame_t + 1:
                    t_indices.append(frame_map[frame_t])
                    t1_indices.append(frame_map[frame_t1])

        # 2-3. 유효한 쌍이 있을 경우에만 Temporal Loss를 계산합니다.
        if t_indices and t1_indices:
            def split_dict(original_dict, indices):
                new_dict = {}
                for key, value in original_dict.items():
                    if isinstance(value, torch.Tensor):
                        new_dict[key] = value[indices]
                    elif isinstance(value, list):
                        new_dict[key] = [value[i] for i in indices]
                return new_dict

            # t와 t+1 데이터가 완벽하게 짝을 이룬 새로운 배치를 만듭니다.
            data_t = split_dict(temporal_batch, t_indices)
            data_t1 = split_dict(temporal_batch, t1_indices)

            out_t = self(data_t)
            out_t1 = self(data_t1)
            
            temporal_loss = compute_temporal_loss(out_t, out_t1, data_t, data_t1)

        # --- 3. 최종 Loss 계산 및 로깅 ---
        alpha = 0.1
        total_loss = main_loss + alpha * temporal_loss

        loss_dict["total_loss"] = total_loss.item()
        loss_dict["main_loss"] = main_loss.item()
        loss_dict["temporal_loss"] = temporal_loss.item()
        
        batch_size_for_log = main_batch["x_positions"].size(0)
        for k, v in loss_dict.items():
            self.log(
                f"train/{k}", v,
                on_step=True, on_epoch=True, prog_bar=True,
                sync_dist=True, batch_size=batch_size_for_log
            )

        return total_loss
    
    def validation_step(self, data, batch_idx):
        if isinstance(data, list):
            data = data[-1]
        out = self(data)
        _, loss_dict = self.cal_loss(out, data)
        metrics = self.val_metrics(out, data['target'][:, 0])
        if out['new_y_hat'] is not None:
            out['y_hat'] = out['new_y_hat']
            out['pi'] = out['new_pi']
        if out['new_y_hat'] is not None:
            metrics_new = self.val_metrics_new(out, data['target'][:, 0])

        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        if out['new_y_hat'] is not None:
            self.log_dict(
                metrics_new,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=data["x_positions"].size(0),
                sync_dist=True,
            )

    def on_test_start(self) -> None:
        save_dir = Path("./submission")
        save_dir.mkdir(exist_ok=True)
        self.submission_handler = SubmissionAv2(save_dir=save_dir)

        # ⏱️ inference time 기록용 변수 초기화
        self.inference_times = []
    
    def test_step(self, data, batch_idx) -> None:
        if isinstance(data, list):
            data = data[-1]

        # ⏱️ 시작 시간
        start_time = time.time()
        out = self(data)
        # ⏱️ 끝 시간
        end_time = time.time()

        # ⏱️ 걸린 시간 저장
        self.inference_times.append(end_time - start_time)

        if out['new_y_hat'] is not None:
            out['y_hat'] = out['new_y_hat']
            out['pi'] = out['new_pi']

        self.submission_handler.format_data(data, out["y_hat"], out["pi"])

    def on_test_end(self) -> None:
        self.submission_handler.generate_submission_file()

        # ⏱️ 평균 / 총 추론 시간 출력
        total_time = sum(self.inference_times)
        avg_time = total_time / len(self.inference_times)
        print(f"[Inference Time] Total: {total_time:.4f}s | Avg per batch: {avg_time:.4f}s")
        
    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-5,
            warmup_epochs=self.warmup_epochs,
            epochs=self.epochs,
        )
        return [optimizer], [scheduler]
