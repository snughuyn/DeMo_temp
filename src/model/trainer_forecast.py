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
from src.utils.submission_ETRI import SubmissionETRI
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
        self.submission_handler = SubmissionETRI()

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
    
    def _as_tensor_on_device(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return torch.tensor(float(x), device=self.device, dtype=torch.float32)
    
    def on_train_epoch_start(self):
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None and hasattr(dm, "pair_batch_sampler") and dm.pair_batch_sampler is not None:
            dm.pair_batch_sampler.set_epoch(self.current_epoch)
        
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

        pred_dense_trajs = forward_ret["pred_dense_trajs"]              # (B, N, T, 7)
        obj_trajs_future_state = forward_ret["obj_trajs_future_state"]  # (B, N, T, 4)
        obj_trajs_future_mask = forward_ret["obj_trajs_future_mask"]    # (B, N, T)

        # --- type 정보 가져오기 ---
        if "x_attr" not in forward_ret:
            return torch.tensor(0.0, device=self.device), tb_dict or {}, disp_dict or {}

        x_attr = forward_ret["x_attr"]  # (B, N, 3)
        agent_type = x_attr[..., 1]     # (B, N)
        mask_valid_agent = (agent_type != 0)  # type==0 (전부 미관측 agent) 제외

        B, N, T, _ = pred_dense_trajs.shape

        # --- 유효 trajectory가 있는 agent만 필터링 ---
        has_future = (obj_trajs_future_mask.sum(dim=-1) > 0)  # (B, N)
        valid_mask = mask_valid_agent & has_future             # (B, N)

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device), tb_dict or {}, disp_dict or {}

        # --- valid agent만 선택 ---
        pred_dense_trajs = pred_dense_trajs[valid_mask]          # (V, T, 7)
        obj_trajs_future_state = obj_trajs_future_state[valid_mask]  # (V, T, 4)
        obj_trajs_future_mask = obj_trajs_future_mask[valid_mask]    # (V, T)

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

    def cal_loss(self, out, data, tag=''):
        # ===== GT 생성 및 저장 =====
        with torch.no_grad():
            gt_pos = data["target"]
            gt_mask = data["target_mask"].bool()
            dt = 0.1
            vel = gt_pos.diff(dim=2, prepend=gt_pos[:, :, :1, :]) / dt
            obj_trajs_future_state = torch.cat([gt_pos, vel], dim=-1)
            obj_trajs_future_mask = gt_mask

        # ===== type==0 (미관측 agent) 제외용 마스크 =====
        x_attr = data["x_attr"]
        agent_type = x_attr[..., 1]
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

        # ===== Dense Prediction Loss =====
        dense_reg_loss = torch.tensor(0.0, device=y.device)
        if dense_predict is not None:
            tgt_mask = data["target_mask"][:, 0].unsqueeze(-1).float()
            dense_reg = F.smooth_l1_loss(dense_predict, y[..., :2], reduction="none")
            denom = tgt_mask.sum().clamp_min(1.0)
            dense_reg_loss = (dense_reg * tgt_mask).sum() / denom
            dense_reg_loss = torch.nan_to_num(dense_reg_loss, nan=0.0)

        # ===== Mode Query Loss (ego용) =====
        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]
        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
        agent_cls_loss = F.cross_entropy(pi, best_mode.detach(), label_smoothing=0.2)

        # ===== Refine Output Loss (ego용) =====
        if new_y_hat is not None:
            l2_norm_new = torch.norm(new_y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
            best_mode_new = torch.argmin(l2_norm_new, dim=-1)
            new_y_hat_best = new_y_hat[torch.arange(new_y_hat.shape[0]), best_mode_new]
            new_agent_reg_loss = F.smooth_l1_loss(new_y_hat_best[..., :2], y)
            new_pi_reg_loss = F.cross_entropy(new_pi, best_mode_new.detach(), label_smoothing=0.2)
        else:
            new_agent_reg_loss = torch.tensor(0.0, device=y.device)
            new_pi_reg_loss = torch.tensor(0.0, device=y.device)

        # ===== Others Loss (type!=0만 포함) =====
        if others_reg_mask is not None and others_reg_mask.sum() > 0:
            others_reg_loss = F.smooth_l1_loss(y_hat_others[others_reg_mask], y_others[others_reg_mask])
        else:
            others_reg_loss = torch.tensor(0.0, device=y.device)

        # ===== Laplace Loss (ego만) =====
        predictions = {"traj": y_hat, "scale": scal, "probs": pi}
        laplace_loss = self.laplace_loss.compute(predictions, y)
        predictions = {"traj": new_y_hat, "scale": scal_new, "probs": new_pi}
        laplace_loss_new = self.laplace_loss.compute(predictions, y)

        # ===== Total =====
        loss = (
            agent_reg_loss + agent_cls_loss +
            others_reg_loss +
            new_agent_reg_loss + new_pi_reg_loss +
            laplace_loss + laplace_loss_new + dense_reg_loss
        )

        # ===== DFP Loss 추가 =====
        dfp_loss, tb_dict, dfp_disp = self.get_dense_future_prediction_loss()
        loss = loss + 1e-5 * dfp_loss

        # ===== Logging =====
        disp_dict = {
            f"{tag}loss": loss.item(),
            f"{tag}reg_loss": agent_reg_loss.item(),
            f"{tag}cls_loss": agent_cls_loss.item(),
            f"{tag}others_reg_loss": others_reg_loss.item(),
            f"{tag}laplace_loss": laplace_loss.item(),
            f"{tag}laplace_loss_new": laplace_loss_new.item(),
            f"{tag}dfp_loss": 1e-5 * dfp_loss.item(),
        }
        return loss, disp_dict

    def training_step(self, batch, batch_idx):
        # ✅ temporal은 None일 수 있음(CombinedLoader max_size)
        main_batch = batch['main']
        temporal_batch = batch.get('temporal', None)

        # ========== 1) Main Loss ==========
        out_main = self(main_batch)
        main_loss, loss_dict = self.cal_loss(out_main, main_batch)

        # ========== 2) Temporal Loss ==========
        # ✅ 안전 초기화
        device = self.device
        temporal_loss = torch.tensor(0.0, device=device)
        num_matched = torch.tensor(0.0, device=device)
        num_total   = torch.tensor(0.0, device=device)

        if (
            temporal_batch is not None
            and isinstance(temporal_batch, dict)
            and 'scenario_id' in temporal_batch
            and 'track_id'   in temporal_batch
            and len(temporal_batch['scenario_id']) > 0
        ):
            # --- (t, t+1) 쌍 재구성 ---
            agents_map = defaultdict(dict)
            # 일부 로더는 list로 scenario_id를 줄 수 있으니 range(len(...))로 순회
            n_agents = len(temporal_batch['scenario_id'])
            for i in range(n_agents):
                sc_id = temporal_batch['scenario_id'][i]
                tk_id = temporal_batch['track_id'][i]
                try:
                    base_id, frame_str = sc_id.rsplit('_', 1)
                    frame = int(frame_str)
                    unique_key = (base_id, tk_id)
                    agents_map[unique_key][frame] = i
                except (ValueError, IndexError):
                    continue

            t_indices, t1_indices = [], []
            for _, frame_map in agents_map.items():
                if len(frame_map) == 2:
                    frame_t  = min(frame_map.keys())
                    frame_t1 = max(frame_map.keys())
                    if frame_t1 == frame_t + 1:
                        t_indices.append(frame_map[frame_t])
                        t1_indices.append(frame_map[frame_t1])

            if t_indices and t1_indices:
                def split_dict(original_dict, indices):
                    new_dict = {}
                    for key, value in original_dict.items():
                        if isinstance(value, torch.Tensor):
                            new_dict[key] = value[indices]
                        elif isinstance(value, list):
                            new_dict[key] = [value[i] for i in indices]
                    return new_dict

                data_t  = split_dict(temporal_batch, t_indices)
                data_t1 = split_dict(temporal_batch, t1_indices)

                # t+1은 고정(no_grad), t는 그래디언트 전파
                with torch.no_grad():
                    out_t1 = self(data_t1)
                out_t = self(data_t)

                # ◽ 과거/현재 버전 호환: (loss, num_matched, num_total) 또는 loss 단독
                ret = compute_temporal_loss(out_t, out_t1, data_t, data_t1, return_counts=True)
                if isinstance(ret, tuple) and len(ret) >= 3:
                    tl, nm, nt = ret
                    temporal_loss = tl if isinstance(tl, torch.Tensor) else torch.tensor(float(tl), device=device)
                    num_matched   = nm if isinstance(nm, torch.Tensor) else torch.tensor(float(nm), device=device)
                    num_total     = nt if isinstance(nt, torch.Tensor) else torch.tensor(float(nt), device=device)
                else:
                    temporal_loss = ret if isinstance(ret, torch.Tensor) else torch.tensor(0.0, device=device)
                    num_matched.zero_()
                    num_total.zero_()
            else:
                # ◽ 쌍이 한 개도 없을 때: 그래프 경로 안정화용 더미 no-op
                if hasattr(self.net, "time_decoder"):
                    dummy = []
                    for p in self.net.time_decoder.parameters():
                        if p.requires_grad:
                            # 첫 원소만 살짝 참조 (연산량 최소화)
                            dummy.append(p.view(-1)[:1].sum())
                    if dummy:
                        temporal_loss = temporal_loss + 0.0 * torch.stack(dummy).sum()
        else:
            # ◽ temporal batch 자체가 None인 step: 동일하게 경로 안정화
            if hasattr(self.net, "time_decoder"):
                dummy = []
                for p in self.net.time_decoder.parameters():
                    if p.requires_grad:
                        dummy.append(p.view(-1)[:1].sum())
                if dummy:
                    temporal_loss = temporal_loss + 0.0 * torch.stack(dummy).sum()

        # ========== 3) 합산 및 로깅 ==========
        total_loss = main_loss + 0.2 * temporal_loss

        # ◽ 로깅 값은 모두 Tensor로, 동일 device로 맞춤 (sync_dist 안정)
        def as_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            return torch.tensor(float(x), device=device)

        log_tensors = {
            "total_loss":        as_tensor(total_loss),
            "main_loss":         as_tensor(main_loss),
            "temporal_loss":     as_tensor(temporal_loss),
            "num_matched":       as_tensor(num_matched),
            "num_total":         as_tensor(num_total),
            # 필요하면 아래도 텐서로 넣기
            "loss":              as_tensor(loss_dict.get("loss", main_loss)),
            "reg_loss":          as_tensor(loss_dict.get("reg_loss", 0.0)),
            "cls_loss":          as_tensor(loss_dict.get("cls_loss", 0.0)),
            "others_reg_loss":   as_tensor(loss_dict.get("others_reg_loss", 0.0)),
            "laplace_loss":      as_tensor(loss_dict.get("laplace_loss", 0.0)),
            "laplace_loss_new":  as_tensor(loss_dict.get("laplace_loss_new", 0.0)),
            "dfp_loss":          as_tensor(loss_dict.get("dfp_loss", 0.0)),
        }

        def _as_tensor_on_device(x, device):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            return torch.tensor(float(x), device=device, dtype=torch.float32)
        
        batch_size_for_log = int(main_batch["x_positions"].size(0))
        
        for k, v in log_tensors.items():
            vv = _as_tensor_on_device(v, self.device)
            try:
                self.log(
                    name=f"train/{k}",
                    value=vv,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=(k in {"total_loss", "main_loss", "temporal_loss"}),  # 원하면 표시만 선정
                    sync_dist=True,
                    batch_size=batch_size_for_log,
                )
            except TypeError:
                # 혹시 여전히 문제가 있으면 log_dict로 우회 (value만 넘김)
                self.log_dict(
                    {f"train/{k}": vv},
                    on_step=True,
                    on_epoch=True,
                    prog_bar=(k in {"total_loss", "main_loss", "temporal_loss"}),
                    sync_dist=True,
                    batch_size=batch_size_for_log,
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
        self.submission_handler = SubmissionETRI(save_dir=save_dir)

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
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }

        base_lr = self.lr
        base_weight_decay = self.weight_decay

        # 💡 그룹별 Learning Rate 설정
        lr_lane_mode = 5e-5      # Lane 관련 모듈 LR
        lr_dfp = 5e-5            # DFP 관련 모듈 LR

        # 💡 그룹 정의
        param_groups = {
            'default': [],
            'lane_mode': [],
            'dfp': [],
        }

        # 💡 키워드 정의
        lane_mode_keywords = ['lane_form_embed', 'lane_query_mlp']
        dfp_keywords = [
            'obj_pos_encoding_layer', 'dense_future_head', 'future_traj_mlps', 
            'traj_fusion_mlps', 'future_embedding', 'future_attn', 
        ]

        # 파라미터 그룹 분류
        for param_name, param in param_dict.items():
            if any(kw in param_name for kw in lane_mode_keywords):
                param_groups['lane_mode'].append(param)
            elif any(kw in param_name for kw in dfp_keywords):
                param_groups['dfp'].append(param)
            else:
                param_groups['default'].append(param)

        # 💡 옵티마이저 그룹 생성
        optimizer_groups = [
            {
                'params': param_groups['default'],
                'lr': base_lr,
                'weight_decay': base_weight_decay,
            },
            {
                'params': param_groups['lane_mode'],
                'lr': lr_lane_mode,
                'weight_decay': base_weight_decay,
            },
            {
                'params': param_groups['dfp'],
                'lr': lr_dfp,
                'weight_decay': base_weight_decay,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_groups, lr=base_lr, weight_decay=base_weight_decay)
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=base_lr,
            min_lr=5e-6,
            warmup_epochs=self.warmup_epochs,
            epochs=self.epochs,
        )

        return [optimizer], [scheduler]


# integrate 'DeMo' with 'RealMotion'
class StreamTrainer(Trainer):
    def __init__(self,
                 num_grad_frame=2,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_grad_frame = num_grad_frame
    
    def training_step(self, data, batch_idx):
        total_step = len(data)
        num_grad_frames = min(self.num_grad_frame, total_step)
        num_no_grad_frames = total_step - num_grad_frames

        memory_dict = None
        self.eval()
        with torch.no_grad():
            for i in range(num_no_grad_frames):
                cur_data = data[i]
                cur_data['memory_dict'] = memory_dict
                out = self(cur_data)
                memory_dict = out['memory_dict']
        
        self.train()
        sum_loss = 0
        loss_dict = {}
        for i in range(num_grad_frames):
            cur_data = data[i + num_no_grad_frames]
            cur_data['memory_dict'] = memory_dict
            out = self(cur_data)
            cur_loss, cur_loss_dict = self.cal_loss(out, cur_data, tag=f'step{i + num_no_grad_frames}_')
            loss_dict.update(cur_loss_dict)
            sum_loss += cur_loss
            memory_dict = out['memory_dict']
        loss_dict['loss'] = sum_loss.item()
        for k, v in loss_dict.items():
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        return sum_loss
    
    def validation_step(self, data, batch_idx):
        memory_dict = None
        all_outs = []
        for i in range(len(data)):
            cur_data = data[i]
            if cur_data['x_positions_diff'].size(1) == 1:
                return
            cur_data['memory_dict'] = memory_dict
            out = self(cur_data)
            _, cur_loss_dict = self.cal_loss(out, cur_data, tag=f'step{i}_')
            memory_dict = out['memory_dict']
            all_outs.append(out)
        
        metrics = self.val_metrics(all_outs[-1], data[-1]['target'][:, 0])
        if all_outs[-1]['new_y_hat'] is not None:
            all_outs[-1]['y_hat'] = all_outs[-1]['new_y_hat']
            all_outs[-1]['pi'] = all_outs[-1]['new_pi']
        if all_outs[-1]['new_y_hat'] is not None:
            metrics_new = self.val_metrics_new(all_outs[-1], data[-1]['target'][:, 0])

        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        if all_outs[-1]['new_y_hat'] is not None:
            self.log_dict(
                metrics_new,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )
    
    def test_step(self, data, batch_idx) -> None:
        memory_dict = None
        all_outs = []
        for i in range(len(data)):
            cur_data = data[i]
            cur_data['memory_dict'] = memory_dict
            out = self(cur_data)
            memory_dict = out['memory_dict']
            all_outs.append(out)

        if all_outs[-1]['new_y_hat'] is not None:
            all_outs[-1]['y_hat'] = all_outs[-1]['new_y_hat']
            all_outs[-1]['pi'] = all_outs[-1]['new_pi']

        self.submission_handler.format_data(data[-1], all_outs[-1]["y_hat"], all_outs[-1]["pi"])
