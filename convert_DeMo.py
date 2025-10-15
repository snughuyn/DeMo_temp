import argparse
import pickle
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm


def resample_polyline(points: np.ndarray, num_interp: int = 20):
    if len(points) < 2:
        return np.zeros((num_interp, 2), dtype=np.float32)

    seg_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_length = np.sum(seg_lengths)
    if total_length < 1e-6:
        return np.tile(points[0], (num_interp, 1))

    cumlen = np.insert(np.cumsum(seg_lengths), 0, 0.0)
    target_dist = np.linspace(0, total_length, num_interp)

    resampled = []
    for d in target_dist:
        idx = np.searchsorted(cumlen, d) - 1
        idx = min(idx, len(points) - 2)
        t = (d - cumlen[idx]) / (cumlen[idx + 1] - cumlen[idx] + 1e-8)
        p = (1 - t) * points[idx] + t * points[idx + 1]
        resampled.append(p)

    return np.array(resampled, dtype=np.float32)


def convert_custom_pkl_to_pt(pkl_file: Path, save_dir: Path,
                             num_past: int = 20, num_future: int = 60, lane_interp: int = 20):
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    log_id = data["log_id"]
    frame_idx = data["frm_idx"]
    agent_data = data["agent"]
    map_data = data["map"]

    num_nodes = agent_data["num_nodes"]
    num_steps = num_past + num_future   # 총 step 수 (80)

    # === Actor features ===
    x_positions = torch.zeros(num_nodes, num_steps, 2)
    x_angles    = torch.zeros(num_nodes, num_steps)
    x_velocity  = torch.zeros(num_nodes, num_steps)
    x_attr      = torch.zeros(num_nodes, 3, dtype=torch.long)
    valid_mask  = torch.ones(num_nodes, num_steps, dtype=torch.bool)  # 기본은 invalid

    agent_ids = agent_data["id"]

    for i in range(num_nodes):
        # 실제 step 길이
        pos = np.array(agent_data["position"][i], dtype=np.float32)
        step_len = pos.shape[0]   # 이 actor가 가진 실제 step 수 (<= num_steps)

        # 위치
        x_positions[i, :step_len] = torch.tensor(pos[:, :2])
        # heading
        if "heading" in agent_data:
            heading = np.array(agent_data["heading"][i], dtype=np.float32)
            x_angles[i, :step_len] = torch.tensor(heading)
        # velocity
        if "velocity" in agent_data:
            vel = np.array(agent_data["velocity"][i], dtype=np.float32)
            vel_norm = np.linalg.norm(vel[:, :2], axis=1)
            x_velocity[i, :step_len] = torch.tensor(vel_norm)

        # valid mask (유효한 구간만 False)
        valid_mask[i, :step_len] = False

        # attribute
        obj_type = int(agent_data.get("type", [0]*num_nodes)[i])
        obj_cat  = int(agent_data.get("category", [0]*num_nodes)[i]) if "category" in agent_data else 0
        combined = obj_type
        x_attr[i] = torch.tensor([obj_type, obj_cat, combined])

    # === Lane features ===
    lane_positions, lane_attr, is_intersections = [], [], []
    for lane in map_data:
        raw_pts = np.array(lane["Pts"], dtype=np.float32)[:, :2]
        centerline = resample_polyline(raw_pts, num_interp=lane_interp)

        width = 3.5
        lane_type = int(lane.get("Type", 0))
        is_inter = 0

        lane_positions.append(torch.tensor(centerline, dtype=torch.float))
        lane_attr.append(torch.tensor([lane_type, width, is_inter], dtype=torch.float))
        is_intersections.append(is_inter)

    lane_positions = torch.stack(lane_positions)          # (num_lanes, lane_interp, 2)
    lane_attr = torch.stack(lane_attr)                    # (num_lanes, 3)
    is_intersections = torch.tensor(is_intersections, dtype=torch.float)

    focal_idx = agent_data.get("av_index", -1)
    scored_idx = [i for i, t in enumerate(agent_data.get("category", [0]*num_nodes)) if t == 2]

    save_dict = {
        "x_positions": x_positions,
        "x_attr": x_attr,
        "x_angles": x_angles,
        "x_velocity": x_velocity,
        "x_valid_mask": ~valid_mask,  # True = 유효, False = 패딩
        "lane_positions": lane_positions,
        "lane_attr": lane_attr,
        "is_intersections": is_intersections,
        "scenario_id": f"{log_id}_{frame_idx}",
        "agent_ids": agent_ids,
        "focal_idx": focal_idx,
        "scored_idx": scored_idx,
        "city": "custom_city",
    }

    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{log_id}_{frame_idx}.pt"
    torch.save(save_dict, save_path)


def main():
    parser = argparse.ArgumentParser(description="Convert QCNet .pkl dataset into DeMo .pt format (valid_mask fixed)")
    parser.add_argument("--data_root", "-d", type=str, required=True)
    parser.add_argument("--save_dir", "-s", type=str, default="data/DeMo_processed")
    parser.add_argument("--num_past", type=int, default=20)
    parser.add_argument("--num_future", type=int, default=60)
    parser.add_argument("--lane_interp", type=int, default=20)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    save_dir = Path(args.save_dir)
    pkl_files = sorted(data_root.rglob("*.pkl"))

    print(f"Found {len(pkl_files)} .pkl files in {data_root}")

    for pkl_file in tqdm(pkl_files, desc="Converting"):
        try:
            convert_custom_pkl_to_pt(pkl_file, save_dir, args.num_past, args.num_future, args.lane_interp)
        except Exception as e:
            print(f"❌ Error while converting {pkl_file}: {e}")

    print(f"✅ Conversion finished. Saved to {save_dir}")


if __name__ == "__main__":
    main()
