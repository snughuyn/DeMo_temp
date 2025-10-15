# src/utils/temporal_consistency.py

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict

def _bidirectional_matching(P_end: torch.Tensor, Q_end: torch.Tensor) -> List[Tuple[int, int]]:
    """ FDE 기반으로 상호 가장 가까운 이웃(MNN) 쌍을 찾습니다. """
    K = P_end.size(0)
    cost_matrix = torch.cdist(P_end, Q_end)

    _, p_to_q_best_idx = cost_matrix.min(dim=1)
    _, q_to_p_best_idx = cost_matrix.min(dim=0)

    mutual_pairs = []
    for k in range(K):
        m = p_to_q_best_idx[k].item()
        if q_to_p_best_idx[m].item() == k:
            mutual_pairs.append((k, m))
    return mutual_pairs

def compute_temporal_loss(
    out_t: Dict[str, torch.Tensor],
    out_t1: Dict[str, torch.Tensor],
    data_t: Dict,
    data_t1: Dict,
    time_shift: int = 1,
) -> torch.Tensor:

    y_hat_t, track_ids_t = out_t["y_hat"], data_t["track_id"]
    y_hat_t1, track_ids_t1 = out_t1["y_hat"], data_t1["track_id"]
    
    device = y_hat_t.device
    B_t, K, T, _ = y_hat_t.shape
    s = time_shift
    total_loss = 0.0
    num_matched_agents = 0

    # 좌표 변환에 필요한 정보
    origin_t, theta_t = data_t["origin"], data_t["theta"]
    origin_t1, theta_t1 = data_t1["origin"], data_t1["theta"]

    track_id_to_idx_t1 = {tid: i for i, tid in enumerate(track_ids_t1)}

    for i, track_id in enumerate(track_ids_t):
        if track_id in track_id_to_idx_t1:
            j = track_id_to_idx_t1[track_id]
            
            P = y_hat_t[i]  # t 예측 (K, T, 2), t의 로컬 좌표계
            Q = y_hat_t1[j] # t+1 예측 (K, T, 2), t+1의 로컬 좌표계

            # --- 🚀 좌표 변환 로직 🚀 ---
            # 1. t+1의 로컬 좌표 -> 글로벌 좌표로 변환
            # 역회전 행렬을 만듭니다. (theta_t1[j] 만큼 회전된 것을 되돌림)
            rot_mat_t1_inv = torch.tensor([
                [torch.cos(-theta_t1[j]), -torch.sin(-theta_t1[j])],
                [torch.sin(-theta_t1[j]), torch.cos(-theta_t1[j])]
            ], device=device, dtype=torch.double)
            Q_global = torch.matmul(Q.double(), rot_mat_t1_inv) + origin_t1[j].double()

            # 2. 글로벌 좌표 -> t의 로컬 좌표로 변환
            # t 시점의 회전 행렬을 만듭니다.
            rot_mat_t = torch.tensor([
                [torch.cos(theta_t[i]), -torch.sin(theta_t[i])],
                [torch.sin(theta_t[i]), torch.cos(theta_t[i])]
            ], device=device, dtype=torch.double)
            Q_transformed = torch.matmul(Q_global - origin_t[i].double(), rot_mat_t).float()
            # --- 좌표 변환 끝 ---

            # 겹치는 구간 추출
            P_overlap = P[:, s:, :]
            Q_overlap = Q_transformed[:, :-s, :] # ✨ 변환된 궤적 사용
            
            P_end = P_overlap[:, -1, :]
            Q_end = Q_overlap[:, -1, :]
            
            matched_pairs = _bidirectional_matching(P_end, Q_end)

            if not matched_pairs:
                continue

            p_matched = torch.stack([P_overlap[k] for (k, _) in matched_pairs])
            q_matched = torch.stack([Q_overlap[m] for (_, m) in matched_pairs])
            
            # ✅ reduction 방식을 'sum'에서 'mean'으로 변경하여 스케일 안정화
            total_loss += F.smooth_l1_loss(p_matched, q_matched, reduction="mean")
            num_matched_agents += 1

    if num_matched_agents == 0:
        return torch.tensor(0.0, device=device)

    return total_loss / num_matched_agents