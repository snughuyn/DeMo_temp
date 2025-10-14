# analysis_stationary_all.py
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

# ====== 설정 ======
SAMPLE_DT = 0.1  # 초/스텝 (데이터가 10Hz라면 0.1, 5Hz면 0.2로 바꾸세요)
OUTPUT_DIR = "results_t_stationary"
OUTPUT_CSV = "stationary_results.csv"
# ==================

def classify_stationary_from_past(pos_xy, past_mask, threshold=0.5):
    """
    과거 궤적만 가지고 stationary 여부 판단.
    pos_xy: (T, 2)
    past_mask: (T,) bool
    threshold: 평균 속도 임계값 (m/s)
    return:
        -1 : invalid (과거 데이터 부족)
         0 : stationary
         1 : moving
    """
    past_traj = pos_xy[past_mask]
    if len(past_traj) < 2:
        return -1  # invalid

    disp = np.linalg.norm(past_traj[-1] - past_traj[0])
    duration = (len(past_traj) - 1) * SAMPLE_DT
    avg_speed = disp / duration if duration > 0 else 0.0

    return 0 if avg_speed < threshold else 1


def analyze_agents(agent_dict):
    """
    각 PKL의 agent dict에서 category == 2 (target agent)만 분석.
    """
    results = []

    positions = agent_dict["position"]      # (N, T, 3)
    predict_mask = agent_dict["predict_mask"]  # (N, T) bool
    valid_mask   = agent_dict["valid_mask"]    # (N, T) bool
    types = agent_dict["type"]              # (N,)
    categories = agent_dict["category"]     # (N,)

    num_agents, T, _ = positions.shape

    for i in range(num_agents):
        if categories[i] != 2:  # 🎯 target agent만 선택
            continue

        pm = predict_mask[i].astype(bool)
        vm = valid_mask[i].astype(bool)
        past_mask = (~pm) & vm

        pos_xy = positions[i, :, :2]

        traj_type = classify_stationary_from_past(pos_xy, past_mask)

        results.append({
            "obj_type": int(types[i]),
            "trajectory_type": traj_type
        })

    return results


TRAJ_TYPE_LABELS = {
    -1: "invalid",
     0: "stationary",
     1: "moving"
}


def process_all_pkls(root_dir, output_csv=OUTPUT_CSV, output_dir=OUTPUT_DIR):
    """
    root_dir 아래 있는 모든 PKL 파일을 처리.
    """
    pkl_files = [f for f in os.listdir(root_dir) if f.endswith(".pkl")]
    all_rows = []

    for pkl_file in tqdm(pkl_files, desc="Processing all PKL files"):
        path = os.path.join(root_dir, pkl_file)
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            if "agent" not in data:
                continue
            rows = analyze_agents(data["agent"])
            if rows:
                df = pd.DataFrame(rows)
                df["source_file"] = pkl_file
                all_rows.append(df)
        except Exception as e:
            print(f"❌ {pkl_file}: {e}")

    if not all_rows:
        print("⚠️ No valid results found.")
        return None

    final_df = pd.concat(all_rows, ignore_index=True)
    final_df.to_csv(output_csv, index=False)

    # type/trajectory type 개수 요약 저장
    os.makedirs(output_dir, exist_ok=True)

    # 1) Object type
    obj_counts = final_df["obj_type"].value_counts().sort_index()
    obj_counts.to_csv(os.path.join(output_dir, "obj_type_counts.csv"), header=["count"])

    # 2) Trajectory type
    traj_counts = final_df["trajectory_type"].value_counts().sort_index()
    traj_counts.index = traj_counts.index.map(lambda x: TRAJ_TYPE_LABELS.get(x, f"unknown_{x}"))
    traj_counts.to_csv(os.path.join(output_dir, "trajectory_type_counts.csv"), header=["count"])

    print("=== SUMMARY ===")
    print("trajectory_type counts (including invalid):")
    print(traj_counts)
    print("\nobj_type counts:")
    print(obj_counts)

    return final_df


if __name__ == "__main__":
    root_dir = "/home/yaaaaaaaang/data/ETRI_TP/raw/train"  # 🔹 test set 경로
    process_all_pkls(root_dir)
