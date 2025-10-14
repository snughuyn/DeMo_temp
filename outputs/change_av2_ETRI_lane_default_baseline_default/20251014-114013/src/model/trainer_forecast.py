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
        agent_type = x_attr[..., 0]     # (B, N)
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
            gt_pos = data["target"]               # (B, N, T, 2)
            gt_mask = data["target_mask"].bool()  # (B, N, T)
            dt = 0.1
            vel = gt_pos.diff(dim=2, prepend=gt_pos[:, :, :1, :]) / dt
            obj_trajs_future_state = torch.cat([gt_pos, vel], dim=-1)
            obj_trajs_future_mask = gt_mask

            if hasattr(self.net, "time_decoder"):
                td = self.net.time_decoder
                if not hasattr(td, "forward_ret_dict"):
                    td.forward_ret_dict = {}
                td.forward_ret_dict["obj_trajs_future_state"] = obj_trajs_future_state
                td.forward_ret_dict["obj_trajs_future_mask"] = obj_trajs_future_mask
                td.forward_ret_dict["x_attr"] = data["x_attr"].detach().clone()

        # ===== type==0 (미관측 agent) 제외용 마스크 =====
        x_attr = data["x_attr"]             # (B, N, 3)
        agent_type = x_attr[..., 0]         # (B, N)
        mask_valid_agent = (agent_type != 0)

        # ===== 예측 결과 =====
        y_hat = out["y_hat"]                     # (B, M, T, 2)
        pi = out["pi"]                           # (B, M)
        y_hat_others = out["y_hat_others"]       # (B, N_others, T, 2)
        scal, scal_new = out["scal"], out["scal_new"]
        new_y_hat = out.get("new_y_hat", None)
        new_pi = out.get("new_pi", None)
        dense_predict = out.get("dense_predict", None)

        # ===== GT =====
        y = data["target"][:, 0]   # ego GT → (B, T, 2)
        y_others = data["target"][:, 1:]  # other agents → (B, N_others, T, 2)

        # === Others (type==0 제외) ===
        if y_hat_others is not None:
            # valid_mask shape 맞춰서 others만 필터링
            valid_mask = mask_valid_agent[:, 1:]  # ego 제외
            y_hat_others = y_hat_others[valid_mask]
            y_others = y_others[valid_mask]
            others_reg_mask = data["target_mask"][:, 1:][valid_mask]
        else:
            others_reg_mask = None

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
        loss = loss + dfp_loss

        # ===== Logging =====
        disp_dict = {
            f"{tag}loss": loss.item(),
            f"{tag}reg_loss": agent_reg_loss.item(),
            f"{tag}cls_loss": agent_cls_loss.item(),
            f"{tag}others_reg_loss": others_reg_loss.item(),
            f"{tag}laplace_loss": laplace_loss.item(),
            f"{tag}laplace_loss_new": laplace_loss_new.item(),
            f"{tag}dfp_loss": dfp_loss.item(),
        }
        return loss, disp_dict

    









    # def get_dense_future_prediction_loss(self, tb_pre_tag='', tb_dict=None, disp_dict=None):

        # if not hasattr(self.net.time_decoder, "forward_ret_dict"):
        #     return torch.tensor(0.0, device=self.device), tb_dict or {}, disp_dict or {}
        # forward_ret = self.net.time_decoder.forward_ret_dict
        # if "pred_dense_trajs" not in forward_ret:
        #     return torch.tensor(0.0, device=self.device), tb_dict or {}, disp_dict or {}
    
        # pred_dense_trajs = forward_ret["pred_dense_trajs"]                    # (B, N, T, 7)
        # obj_trajs_future_state = forward_ret["obj_trajs_future_state"]        # (B, N, T, 4)
        # obj_trajs_future_mask = forward_ret["obj_trajs_future_mask"]          # (B, N, T)
    
        # assert pred_dense_trajs.shape[-1] == 7
        # assert obj_trajs_future_state.shape[-1] == 4
    
        # # === 분리 ===
        # pred_dense_trajs_gmm = pred_dense_trajs[..., 0:5]   # [B, N, T, 5]
        # pred_dense_trajs_vel = pred_dense_trajs[..., 5:7]   # [B, N, T, 2]
    
        # # === velocity loss ===
        # loss_vel = F.l1_loss(pred_dense_trajs_vel, obj_trajs_future_state[..., 2:4], reduction='none')
        # loss_vel = (loss_vel * obj_trajs_future_mask[..., None]).sum(dim=-1).sum(dim=-1)
        # # [B, N]
    
        # B, N, T, _ = pred_dense_trajs.shape
        # fake_scores = pred_dense_trajs.new_zeros((B, N)).view(-1, 1)  # [B*N, 1]
        # temp_pred_trajs = pred_dense_trajs_gmm.contiguous().view(B * N, 1, T, 5)
        # temp_gt_idx = torch.zeros(B * N, dtype=torch.long, device=pred_dense_trajs.device)
        # temp_gt_trajs = obj_trajs_future_state[..., 0:2].contiguous().view(B * N, T, 2)
        # temp_gt_trajs_mask = obj_trajs_future_mask.view(B * N, T)
    
        # # === GMM NLL loss ===
        # loss_gmm, _ = nll_loss_gmm_direct(
        #     pred_scores=fake_scores,
        #     pred_trajs=temp_pred_trajs,
        #     gt_trajs=temp_gt_trajs,
        #     gt_valid_mask=temp_gt_trajs_mask,
        #     pre_nearest_mode_idxs=temp_gt_idx,
        #     timestamp_loss_weight=None,
        #     use_square_gmm=False,
        # )
        # loss_gmm = loss_gmm.view(B, N)
    
        # # === combine velocity + GMM ===
        # loss_reg = loss_gmm + loss_vel
    
        # obj_valid_mask = obj_trajs_future_mask.sum(dim=-1) > 0  # [B, N]
        # loss_reg = (loss_reg * obj_valid_mask.float()).sum(dim=-1) / torch.clamp_min(obj_valid_mask.sum(dim=-1), 1.0)
        # loss_reg = loss_reg.mean()
    
        # if tb_dict is None: tb_dict = {}
        # if disp_dict is None: disp_dict = {}
    
        # tb_dict[f'{tb_pre_tag}loss_dense_prediction'] = loss_reg.item()
        # disp_dict[f'{tb_pre_tag}loss_dense_prediction'] = loss_reg.item()
    
        # return loss_reg, tb_dict, disp_dict

    # def cal_loss(self, out, data, tag=''):
    #     # ===== DFP용 GT 생성 및 저장 =====
    #     with torch.no_grad():
    #         gt_pos = data["target"]               # (B, N, T, 2)
    #         gt_mask = data["target_mask"].bool()  # (B, N, T)

    #         dt = 0.1
    #         vel = gt_pos.diff(dim=2, prepend=gt_pos[:, :, :1, :]) / dt  # (B, N, T, 2)
    #         obj_trajs_future_state = torch.cat([gt_pos, vel], dim=-1)   # (B, N, T, 4)
    #         obj_trajs_future_mask = gt_mask

    #         if hasattr(self.net, "time_decoder"):
    #             td = self.net.time_decoder
    #             if not hasattr(td, "forward_ret_dict"):
    #                 td.forward_ret_dict = {}
    #             td.forward_ret_dict["obj_trajs_future_state"] = obj_trajs_future_state
    #             td.forward_ret_dict["obj_trajs_future_mask"] = obj_trajs_future_mask

    #     # ===== 기존 예측 및 GT =====
    #     y_hat, pi, y_hat_others = out["y_hat"], out["pi"], out["y_hat_others"]
    #     scal, scal_new = out["scal"], out["scal_new"]
    #     new_y_hat = out.get("new_y_hat", None)
    #     new_pi = out.get("new_pi", None)
    #     dense_predict = out.get("dense_predict", None)

    #     y, y_others = data["target"][:, 0], data["target"][:, 1:]

    #     # ===== Dense Prediction Loss =====
    #     if dense_predict is not None:
    #         tgt_mask = data["target_mask"][:, 0].unsqueeze(-1).float()
    #         dense_reg = F.smooth_l1_loss(dense_predict, y[..., :2], reduction="none")
        
    #         # 유효 mask가 0인 경우를 방지하고 NaN 제거
    #         denom = tgt_mask.sum().clamp_min(1.0)
    #         dense_reg_loss = (dense_reg * tgt_mask).sum() / denom
    #         dense_reg_loss = torch.nan_to_num(dense_reg_loss, nan=0.0, posinf=1.0, neginf=0.0)
    #     else:
    #         dense_reg_loss = torch.tensor(0.0, device=y.device)
        

    #     # ===== Mode Query Loss =====
    #     l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
    #     best_mode = torch.argmin(l2_norm, dim=-1)
    #     y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]
    #     agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
    #     agent_cls_loss = F.cross_entropy(pi, best_mode.detach(), label_smoothing=0.2)

    #     # ===== Refine Output Loss =====
    #     if new_y_hat is not None:
    #         l2_norm_new = torch.norm(new_y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
    #         best_mode_new = torch.argmin(l2_norm_new, dim=-1)
    #         new_y_hat_best = new_y_hat[torch.arange(new_y_hat.shape[0]), best_mode_new]
    #         new_agent_reg_loss = F.smooth_l1_loss(new_y_hat_best[..., :2], y)
    #     else:
    #         new_agent_reg_loss = 0.0

    #     new_pi_reg_loss = 0.0 if new_pi is None else F.cross_entropy(new_pi, best_mode_new.detach(), label_smoothing=0.2)

    #     # ===== Other Agents Loss =====
    #     others_reg_mask = data["target_mask"][:, 1:]
    #     others_reg_loss = F.smooth_l1_loss(y_hat_others[others_reg_mask], y_others[others_reg_mask])

    #     # ===== Laplace Loss =====
    #     predictions = {"traj": y_hat, "scale": scal, "probs": pi}
    #     laplace_loss = self.laplace_loss.compute(predictions, y)
    #     predictions = {"traj": new_y_hat, "scale": scal_new, "probs": new_pi}
    #     laplace_loss_new = self.laplace_loss.compute(predictions, y)

    #     # ===== Total Basic Loss =====
    #     loss = (agent_reg_loss + agent_cls_loss + others_reg_loss +
    #             new_agent_reg_loss + dense_reg_loss + new_pi_reg_loss +
    #             laplace_loss + laplace_loss_new)

    #     # ===== DFP Loss 추가 =====
    #     dfp_loss, tb_dict, dfp_disp = self.get_dense_future_prediction_loss()
    #     loss = loss + 1e-4 * dfp_loss

    #     # ===== Logging Dictionary =====
    #     disp_dict = {
    #         f"{tag}loss": loss.item(),
    #         f"{tag}reg_loss": agent_reg_loss.item(),
    #         f"{tag}cls_loss": agent_cls_loss.item(),
    #         f"{tag}others_reg_loss": others_reg_loss.item(),
    #         f"{tag}laplace_loss": laplace_loss.item(),
    #         f"{tag}laplace_loss_new": laplace_loss_new.item(),
    #         f"{tag}dfp_loss": dfp_loss.item(),
    #     }
    #     if new_y_hat is not None:
    #         disp_dict[f"{tag}reg_loss_refine"] = new_agent_reg_loss.item()
    #     if new_pi is not None:
    #         disp_dict[f"{tag}reg_loss_new_pi"] = new_pi_reg_loss.item()
    #     if dense_predict is not None:
    #         disp_dict[f"{tag}reg_loss_dense"] = dense_reg_loss.item()

    #     disp_dict.update(dfp_disp)
    #     return loss, disp_dict



    def training_step(self, data, batch_idx):
        if isinstance(data, list):
            data = data[-1]
        out = self(data)
        loss, loss_dict = self.cal_loss(out, data)

        for k, v in loss_dict.items():
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=data["x_positions"].size(0),  #add
            )

        return loss

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
