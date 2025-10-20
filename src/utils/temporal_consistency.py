import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict

def _bidirectional_matching(P_end: torch.Tensor, Q_end: torch.Tensor) -> List[Tuple[int, int]]:
    cost_matrix = torch.cdist(P_end, Q_end)
    _, p2q = cost_matrix.min(dim=1)
    _, q2p = cost_matrix.min(dim=0)

    mutual_pairs = []
    for k in range(P_end.size(0)):
        m = p2q[k].item()
        if q2p[m].item() == k:
            mutual_pairs.append((k, m))
    return mutual_pairs


@torch.no_grad()
def compute_temporal_loss(
    out_t: Dict[str, torch.Tensor],
    out_t1: Dict[str, torch.Tensor],
    data_t: Dict,
    data_t1: Dict,
    time_shift: int = 1,
    return_counts: bool = False,
) -> torch.Tensor:
    device = out_t["y_hat"].device
    s = time_shift
    total_loss = torch.tensor(0.0, device=device)
    num_matched_agents = 0
    num_total = 0

    # --- 데이터 추출 ---
    y_hat_t  = out_t["y_hat"].float()
    y_hat_t1 = out_t1["y_hat"].float().detach()
    origin_t  = data_t["origin"].float()
    origin_t1 = data_t1["origin"].float()
    theta_t  = data_t["theta"].float()
    theta_t1 = data_t1["theta"].float()
    track_ids_t, track_ids_t1 = data_t["track_id"], data_t1["track_id"]

    # track_id → index 매핑
    t1_map = {tid: j for j, tid in enumerate(track_ids_t1)}

    def local_to_global(y, origin, theta):
        c, s_ = torch.cos(theta), torch.sin(theta)
        rot = torch.stack([
            torch.stack([c, -s_]),
            torch.stack([s_,  c])
        ], dim=0)  # (2, 2)
        return torch.matmul(y, rot.T) + origin  # (K, T, 2)

    for i, tid in enumerate(track_ids_t):
        if tid not in t1_map:
            continue
        j = t1_map[tid]
        num_total += 1

        P = y_hat_t[i]   # (K, T, 2)
        Q = y_hat_t1[j]  # (K, T, 2)

        # ===== (1) 좌표계 통일: local → global 변환 =====
        P_global = local_to_global(P, origin_t[i], theta_t[i])
        Q_global = local_to_global(Q, origin_t1[j], theta_t1[j])

        # ===== (2) Overlap 구간 추출 =====
        P_overlap = P_global[:, s:, :]
        Q_overlap = Q_global[:, :-s, :]

        # ===== (3) Bidirectional FDE 매칭 =====
        P_end, Q_end = P_overlap[:, -1, :], Q_overlap[:, -1, :]
        pairs = _bidirectional_matching(P_end, Q_end)
        if not pairs:
            continue

        p_sel = torch.stack([P_overlap[p] for (p, _) in pairs])
        q_sel = torch.stack([Q_overlap[q] for (_, q) in pairs])

        # ===== (4) Smooth L1 + clipping =====
        pair_loss = F.smooth_l1_loss(p_sel, q_sel, reduction="none").mean(dim=[1, 2])
        pair_loss = torch.clamp(pair_loss, max=10.0)
        total_loss += pair_loss.mean()
        num_matched_agents += 1

    if num_matched_agents > 0:
        loss = total_loss / num_matched_agents
    else:
        loss = torch.tensor(0.0, device=device)

    if return_counts:
        return loss, num_matched_agents, num_total
    return loss
