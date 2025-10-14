import torch
import torch.nn as nn
from .transformer_blocks import Cross_Block, Block
import torch.nn.functional as F
from .mamba.vim_mamba import init_weights, create_block
from functools import partial
from timm.models.layers import DropPath, to_2tuple
from src.utils import build_mlps
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
    


class GMMPredictor_dense(nn.Module):
    def __init__(self, future_len=60, dim=128):
        super(GMMPredictor_dense, self).__init__()
        self._future_len = future_len
        self.gaussian = nn.Sequential(
            nn.Linear(dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 2)
        )
        self.score = nn.Sequential(
            nn.Linear(dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 1),
        )
        self.scale = nn.Sequential(
            nn.Linear(dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 2)
        )
    
    def forward(self, input):
        res = self.gaussian(input)
        scal = F.elu_(self.scale(input), alpha=1.0) + 1.0 + 0.0001
        input = input.max(dim=2)[0]  
        score = self.score(input).squeeze(-1)

        return res, score, scal


class GMMPredictor(nn.Module):
    def __init__(self, future_len=60, dim=128):
        super(GMMPredictor, self).__init__()
        self._future_len = future_len
        self.gaussian = nn.Sequential(
            nn.Linear(dim, 256), 
            nn.GELU(), 
            nn.Linear(256, self._future_len*2)
        )
        self.score = nn.Sequential(
            nn.Linear(dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 1),
        )
        self.scale = nn.Sequential(
            nn.Linear(dim, 256), 
            nn.GELU(), 
            nn.Linear(256, self._future_len*2)
        )
    
    def forward(self, input):
        B, M, _ = input.shape
        res = self.gaussian(input).view(B, M, self._future_len, 2) 
        scal = F.elu(self.scale(input), alpha=1.0) + 1.05  # ✅ inplace 제거 + 살짝 offset 여유
        scal = torch.clamp(scal, min=1e-3, max=50.0) 
        scal = scal.view(B, M, self._future_len, 2) 
        score = self.score(input).squeeze(-1)

        return res, score, scal
    

class TimeDecoder(nn.Module):
    def __init__(self, future_len=60, dim=128):
        super(TimeDecoder, self).__init__()

        ###### State Consistency Module ######
        # state cross attention
        self.cross_block_time = nn.ModuleList(
            Cross_Block()
            for i in range(2)
        )

        # state bidirectional mamba
        self.timequery_embed_mamba = nn.ModuleList(  
            [
                create_block(  
                    d_model=dim,
                    layer_idx=i,
                    drop_path=0.2,  
                    bimamba=True,  
                    rms_norm=True,  
                )
                for i in range(2)
            ]
        )  
        self.timequery_norm_f = RMSNorm(dim, eps=1e-5)
        self.timequery_drop_path = DropPath(0.2)

        # MLP for state query
        self.dense_predict = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, dim),
            nn.GELU(),
            nn.Linear(dim, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

        self.future_embedding = nn.Sequential(
            nn.Linear(7 * future_len, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        self.future_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=8,
            batch_first=True
        )

        ###### Mode Localization Module ######
        # mode self attention
        self.self_block_mode = nn.ModuleList(
            Block()
            for i in range(3)
        )

        # mode cross attention
        self.cross_block_mode = nn.ModuleList(
            Cross_Block()
            for i in range(3)
        )

        # mode query initialization
        self.multi_modal_query_embedding = nn.Embedding(6, dim)
        self.register_buffer('modal', torch.arange(6).long())

        # MLP for mode query
        self.predictor = GMMPredictor(future_len)

        ###### Hybrid Coupling Module ######
        # hybrid self attention
        self.self_block_dense = nn.ModuleList(
            Block()
            for i in range(3)
        )

        # hybrid cross attention
        self.cross_block_dense = nn.ModuleList(
            Cross_Block()
            for i in range(3)
        )

        # mode self attention for hybrid spatiotemporal queries
        self.self_block_different_mode = nn.ModuleList(
            Block()
            for i in range(3)
        )

        # state bidirectional mamba for hybrid spatiotemporal queries 
        self.dense_embed_mamba = nn.ModuleList(  
            [
                create_block(  
                    d_model=dim,
                    layer_idx=i,
                    drop_path=0.2,  
                    bimamba=True,  
                    rms_norm=True,  
                )
                for i in range(2)
            ]
        )
        self.dense_norm_f = RMSNorm(dim, eps=1e-5)
        self.dense_drop_path = DropPath(0.2)

        # MLP for final output
        self.predictor_dense = GMMPredictor_dense(future_len)
        self.build_dfp(hidden_dim=dim, num_future_frames=future_len)

    def build_dfp(self, hidden_dim=128, num_future_frames=60):

        self.num_future_frames = num_future_frames
        self.hidden_dim = hidden_dim

        self.obj_pos_encoding_layer = build_mlps(
            c_in=2, mlp_channels=[hidden_dim, hidden_dim, hidden_dim],
            ret_before_act=True, without_norm=True
        )

        self.dense_future_head = build_mlps(
            c_in=hidden_dim * 2,
            mlp_channels=[hidden_dim, hidden_dim, num_future_frames * 7],
            ret_before_act=True
        )

        self.future_traj_mlps = build_mlps(
            c_in=4 * num_future_frames,
            mlp_channels=[hidden_dim, hidden_dim, hidden_dim],
            ret_before_act=True, without_norm=True
        )

        self.traj_fusion_mlps = build_mlps(
            c_in=hidden_dim * 2,
            mlp_channels=[hidden_dim, hidden_dim, hidden_dim],
            ret_before_act=True, without_norm=True
        )

    def apply_dfp(self, encoding, x_attr, obj_mask, obj_pos):
        """
        encoding: (B, N_all, C) - actor + lane 포함
        x_attr: (B, N_actor, 3)
        obj_mask: (B, N_actor)
        obj_pos: (B, N_actor, 2)
        """

        agent_type = x_attr[..., 0]  # (B, N_actor)
        B, N_all, C = encoding.shape
        N_actor = agent_type.size(1)

        # ✅ actor 부분만 추출 (lane 제외)
        encoding_actor = encoding[:, :N_actor, :]  # (B, N_actor, C)

        device = encoding.device
        num_future_frames = self.num_future_frames

        # === type이 0이 아닌 객체만 DFP 적용 ===
        mask_dfp = (agent_type != 0) & obj_mask  # (B, N_actor)
        valid_count = mask_dfp.sum().item()

        if valid_count == 0:
            # type≠0인 객체가 하나도 없으면 스킵
            if not hasattr(self, "forward_ret_dict"):
                self.forward_ret_dict = {}
            self.forward_ret_dict['pred_dense_trajs'] = torch.zeros(
                B, N_actor, num_future_frames, 7, device=device
            )
            return encoding, self.forward_ret_dict['pred_dense_trajs']

        # === DFP 적용할 valid 객체만 추출 ===
        obj_pos_valid = obj_pos[mask_dfp][..., :2]                 # (V, 2)
        obj_feature_valid = encoding_actor[mask_dfp]               # (V, C)
        obj_pos_feature_valid = self.obj_pos_encoding_layer(obj_pos_valid)  # (V, C)

        obj_fused_feature_valid = torch.cat(
            (obj_pos_feature_valid, obj_feature_valid), dim=-1
        )  # (V, 2C)

        # ✅ BatchNorm-safe 실행: 샘플이 1개일 때 eval 모드로 전환
        was_training = self.training
        if valid_count == 1:
            self.eval()
            with torch.no_grad():
                pred_dense_trajs_valid = self.dense_future_head(obj_fused_feature_valid)
            if was_training:
                self.train()
        else:
            pred_dense_trajs_valid = self.dense_future_head(obj_fused_feature_valid)  # (V, 420)

        pred_dense_trajs_valid = pred_dense_trajs_valid.view(-1, num_future_frames, 7)  # (V, T, 7)

        # === 예측 좌표를 실제 위치 기준으로 보정 ===
        temp_center = pred_dense_trajs_valid[:, :, :2] + obj_pos_valid[:, None, :2]
        pred_dense_trajs_valid = torch.cat(
            (temp_center, pred_dense_trajs_valid[:, :, 2:]), dim=-1
        )  # (V, 60, 7)

        # === trajectory feature 융합 ===
        obj_future_input_valid = pred_dense_trajs_valid[:, :, [0, 1, -2, -1]].flatten(start_dim=1, end_dim=2)
        obj_future_feature_valid = self.future_traj_mlps(obj_future_input_valid)

        obj_full_trajs_feature = torch.cat(
            (obj_feature_valid, obj_future_feature_valid), dim=-1
        )
        obj_feature_valid = self.traj_fusion_mlps(obj_full_trajs_feature)  # (V, C)

        # === actor 부분 업데이트 ===
        ret_obj_feature = torch.zeros_like(encoding_actor)
        ret_obj_feature[mask_dfp] = obj_feature_valid

        # === DFP 예측 결과 저장 ===
        ret_pred_dense_future_trajs = encoding.new_zeros(B, N_actor, num_future_frames, 7)
        ret_pred_dense_future_trajs[mask_dfp] = pred_dense_trajs_valid

        # === forward_ret_dict 갱신 ===
        if not hasattr(self, "forward_ret_dict"):
            self.forward_ret_dict = {}
        self.forward_ret_dict["pred_dense_trajs"] = ret_pred_dense_future_trajs  # (B, N_actor, T, 7)

        # ✅ 안전하게 복사해서 업데이트 (in-place → out-of-place)
        updated_encoding = encoding.clone()                      # 새 텐서 복사
        updated_encoding[:, :N_actor, :] = ret_obj_feature       # actor 부분만 교체
        self.forward_ret_dict["x_attr"] = x_attr.detach().clone()


        return updated_encoding, ret_pred_dense_future_trajs



    # def forward(self, mode, encoding, mask=None, **kwargs):
    #     # ===== Dynamic State Consistency =====
    #     for blk in self.cross_block_time:
    #         mode = blk(mode, encoding, key_padding_mask=mask)

    #     residual = None
    #     for blk_mamba in self.timequery_embed_mamba:
    #         mode, residual = blk_mamba(mode, residual)

    #     fused_add_norm_fn = rms_norm_fn if isinstance(self.timequery_norm_f, RMSNorm) else layer_norm_fn
    #     mode = fused_add_norm_fn(
    #         self.timequery_drop_path(mode),
    #         self.timequery_norm_f.weight,
    #         self.timequery_norm_f.bias,
    #         eps=self.timequery_norm_f.eps,
    #         residual=residual,
    #         prenorm=False,
    #         residual_in_fp32=True
    #     )

    #     dense_pred = self.dense_predict(mode)
    #     mode_tmp = mode

    #     # ===== Mode Localization =====
    #     multi_modal_query = self.multi_modal_query_embedding(self.modal)
    #     mode_query = encoding[:, 0]
    #     mode = mode_query[:, None] + multi_modal_query

    #     for blk in self.cross_block_mode:
    #         mode = blk(mode, encoding, key_padding_mask=mask)
    #     for blk in self.self_block_mode:
    #         mode = blk(mode)

    #     y_hat, pi, scal = self.predictor(mode)

    #     # ===== Hybrid Coupling =====
    #     mode_dense = mode[:, :, None] + mode_tmp[:, None, :]
    #     B, M, T, C = mode_dense.shape

    #     mode_dense = mode_dense.reshape(B, -1, C)
    #     for blk in self.cross_block_dense:
    #         mode_dense = blk(mode_dense, encoding, key_padding_mask=mask)
    #     for blk in self.self_block_dense:
    #         mode_dense = blk(mode_dense)
    #     mode_dense = mode_dense.reshape(B, M, T, C)

    #     mode_dense = mode_dense.transpose(1, 2).reshape(-1, M, C)
    #     for blk in self.self_block_different_mode:
    #         mode_dense = blk(mode_dense)
    #     mode_dense = mode_dense.reshape(B, -1, M, C).transpose(1, 2)

    #     mode_dense = mode_dense.reshape(-1, T, C)
    #     residual = None
    #     for blk_mamba in self.dense_embed_mamba:
    #         mode_dense, residual = blk_mamba(mode_dense, residual)

    #     fused_add_norm_fn = rms_norm_fn if isinstance(self.dense_norm_f, RMSNorm) else layer_norm_fn
    #     mode_dense = fused_add_norm_fn(
    #         self.dense_drop_path(mode_dense),
    #         self.dense_norm_f.weight,
    #         self.dense_norm_f.bias,
    #         eps=self.dense_norm_f.eps,
    #         residual=residual,
    #         prenorm=False,
    #         residual_in_fp32=True
    #     )
    #     mode_dense = mode_dense.reshape(B, M, T, C)

    #     # ===== DFP 적용 (type != 0 대상) =====
    #     x_attr = kwargs.get("x_attr", None)               # (B, N, 3)
    #     x_pos = kwargs.get("x_positions", None)           # (B, N, T, 2)
    #     x_kvm = kwargs.get("x_key_valid_mask", None)      # (B, N)
    #     if x_attr is not None and x_pos is not None:
    #         agent_type = x_attr[..., 0]                   # (B, N)
    #         obj_mask = (agent_type != 0)
    #         if x_kvm is not None:
    #             obj_mask = obj_mask & x_kvm               # type!=0 & valid
    #         last_pos = x_pos[:, :, -1, :]                 # (B, N, 2)

    #         updated_feature, pred_dense_future_trajs = self.apply_dfp(
    #             encoding=encoding,
    #             x_attr=x_attr,
    #             obj_mask=obj_mask,
    #             obj_pos=last_pos
    #         )
    #         encoding = updated_feature

    #         # forward_ret_dict 업데이트
    #         if not hasattr(self, "forward_ret_dict"):
    #             self.forward_ret_dict = {}
    #         self.forward_ret_dict["pred_dense_trajs"] = pred_dense_future_trajs  # (B, N, T, 7)

    #     # ===== Final Prediction =====
    #     y_hat_new, pi_new, scal_new = self.predictor_dense(mode_dense)

    #     return dense_pred, y_hat, pi, mode, y_hat_new, pi_new, mode_dense, scal, scal_new

    def forward(self, mode, encoding, mask=None, **kwargs):
        """
        mode: (B, T, C) - time query embedding
        encoding: (B, N_all, C) - actor + lane feature
        mask: (B, N_all)
        kwargs: includes x_attr, x_positions, x_key_valid_mask
        """
        # ===== Dynamic State Consistency =====
        for blk in self.cross_block_time:
            mode = blk(mode, encoding, key_padding_mask=mask)

        residual = None
        for blk_mamba in self.timequery_embed_mamba:
            mode, residual = blk_mamba(mode, residual)

        fused_add_norm_fn = rms_norm_fn if isinstance(self.timequery_norm_f, RMSNorm) else layer_norm_fn
        mode = fused_add_norm_fn(
            self.timequery_drop_path(mode),
            self.timequery_norm_f.weight,
            self.timequery_norm_f.bias,
            eps=self.timequery_norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True
        )

        dense_pred = self.dense_predict(mode)
        mode_tmp = mode

        # ===== Mode Localization =====
        multi_modal_query = self.multi_modal_query_embedding(self.modal)
        mode_query = encoding[:, 0]  # ego agent feature
        mode = mode_query[:, None] + multi_modal_query  # (B, M, C)

        for blk in self.cross_block_mode:
            mode = blk(mode, encoding, key_padding_mask=mask)
        for blk in self.self_block_mode:
            mode = blk(mode)

        y_hat, pi, scal = self.predictor(mode)

        # ===== Hybrid Coupling =====
        mode_dense = mode[:, :, None] + mode_tmp[:, None, :]
        B, M, T, C = mode_dense.shape

        mode_dense = mode_dense.reshape(B, -1, C)
        for blk in self.cross_block_dense:
            mode_dense = blk(mode_dense, encoding, key_padding_mask=mask)
        for blk in self.self_block_dense:
            mode_dense = blk(mode_dense)
        mode_dense = mode_dense.reshape(B, M, T, C)

        mode_dense = mode_dense.transpose(1, 2).reshape(-1, M, C)
        for blk in self.self_block_different_mode:
            mode_dense = blk(mode_dense)
        mode_dense = mode_dense.reshape(B, -1, M, C).transpose(1, 2)

        mode_dense = mode_dense.reshape(-1, T, C)
        residual = None
        for blk_mamba in self.dense_embed_mamba:
            mode_dense, residual = blk_mamba(mode_dense, residual)

        fused_add_norm_fn = rms_norm_fn if isinstance(self.dense_norm_f, RMSNorm) else layer_norm_fn
        mode_dense = fused_add_norm_fn(
            self.dense_drop_path(mode_dense),
            self.dense_norm_f.weight,
            self.dense_norm_f.bias,
            eps=self.dense_norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True
        )
        mode_dense = mode_dense.reshape(B, M, T, C)

        # ===== DFP 적용 (type != 0 대상) =====
        x_attr = kwargs.get("x_attr", None)               # (B, N, 3)
        x_pos = kwargs.get("x_positions", None)           # (B, N, T, 2)
        x_kvm = kwargs.get("x_key_valid_mask", None)      # (B, N)
        dfp_future_emb = None

        if x_attr is not None and x_pos is not None:
            agent_type = x_attr[..., 0]                   # (B, N)
            obj_mask = (agent_type != 0)
            if x_kvm is not None:
                obj_mask = obj_mask & x_kvm               # type!=0 & valid
            last_pos = x_pos[:, :, -1, :]                 # (B, N, 2)

            updated_feature, pred_dense_future_trajs = self.apply_dfp(
                encoding=encoding,
                x_attr=x_attr,
                obj_mask=obj_mask,
                obj_pos=last_pos
            )
            encoding = updated_feature

            # === DFP 예측 결과를 (B, N, C) 임베딩으로 변환 ===
            dfp_future_emb = self.future_embedding(
                pred_dense_future_trajs.flatten(start_dim=2)
            )  # (B, N, C)

            # forward_ret_dict 업데이트
            if not hasattr(self, "forward_ret_dict"):
                self.forward_ret_dict = {}
            self.forward_ret_dict["pred_dense_trajs"] = pred_dense_future_trajs  # (B, N, T, 7)

        # ===== Future-Aware Mode Update (DFP + Attention) =====
        if dfp_future_emb is not None:
            # mode: (B, M, C), dfp_future_emb: (B, N, C)
            mode, _ = self.future_attn(
                query=mode, key=dfp_future_emb, value=dfp_future_emb
            )

        # ===== Final Prediction =====
        y_hat_new, pi_new, scal_new = self.predictor_dense(mode_dense)

        return dense_pred, y_hat, pi, mode, y_hat_new, pi_new, mode_dense, scal, scal_new
