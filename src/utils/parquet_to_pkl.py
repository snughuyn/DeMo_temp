import os
import torch
import pickle
import pandas as pd
import numpy as np
from pathlib import Path


def parquet_to_pkl_with_dataset(
    parquet_path: str,
    dataset_root: str,
    save_dir: str,
    use_pt: bool = True,
):
    """
    Convert DeMo-style parquet output into QCNet-style multi-agent .pkl
    """
    df = pd.read_parquet(parquet_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for scenario_id, scene_df in df.groupby("scenario_id"):
        # --- 1️⃣ 파일명 구성 ---
        file_name = f"log_{scenario_id}.pt" if use_pt else f"log_{scenario_id}.pkl"
        file_path = Path(dataset_root) / file_name
        if not file_path.exists():
            print(f"[!] Missing scene file: {file_name}, skipping")
            continue

        # --- 2️⃣ Scene 로드 ---
        if use_pt:
            scene_data = torch.load(file_path)
        else:
            with open(file_path, "rb") as f:
                scene_data = pickle.load(f)

        # --- 3️⃣ 기본 정보 ---
        agent_ids = scene_data.get("agent_ids", [])
        x_attr = scene_data.get("x_attr", None)
        scored_idx = scene_data.get("scored_idx", [])
        num_nodes = len(agent_ids)
        num_valid_nodes = len(scored_idx) + 1

        # ✅ category = x_attr[:, 1]
        if isinstance(x_attr, torch.Tensor):
            x_attr = x_attr.cpu().numpy()
        elif isinstance(x_attr, list):
            x_attr = np.array(x_attr)
        if x_attr is not None and x_attr.ndim == 2 and x_attr.shape[1] >= 2:
            categories = x_attr[:, 1].astype(int)
        else:
            categories = np.zeros(num_nodes, dtype=int)

        # --- 4️⃣ multimodal trajectory 수집 ---
        grouped = scene_df.groupby("track_id")
        agent_pred_map = {}
        max_modes = 0

        for tid, agent_df in grouped:
            agent_df = agent_df.sort_values(by="probability", ascending=False).reset_index(drop=True)
            M = len(agent_df)
            H = len(agent_df.iloc[0]["predicted_trajectory_x"])
            trajs = np.zeros((M, H, 2), dtype=np.float32)

            for m, row in enumerate(agent_df.itertuples(index=False)):
                trajs[m, :, 0] = np.array(row.predicted_trajectory_x)
                trajs[m, :, 1] = np.array(row.predicted_trajectory_y)

            agent_pred_map[tid] = trajs
            max_modes = max(max_modes, M)

        predictions = np.full((num_nodes, max_modes, H, 2), -1000.0, dtype=np.float32)

        for tid, trajs in agent_pred_map.items():
            if tid in agent_ids:
                idx = agent_ids.index(tid)
                M = trajs.shape[0]
                predictions[idx, :M] = trajs

        # --- 5️⃣ log_id / frm_idx 분리 ---
        clean_id = scenario_id.replace("_masked", "")
        parts = clean_id.split("_")
        log_id, frm_idx_str = parts[0], parts[1] if len(parts) > 1 else "0"
        frm_idx = int(frm_idx_str)

        # --- 6️⃣ QCNet-style scene 구성 ---
        scene = {
            "log_id": log_id,
            "frm_idx": frm_idx,
            "agent": {
                "num_nodes": num_nodes,
                "num_valid_nodes": num_valid_nodes,
                "id": agent_ids,
                "category": np.array(categories, dtype=int),
                "predictions": predictions,
            },
        }

        # --- 7️⃣ 저장 ---
        save_name = f"log_{log_id}_{frm_idx_str}_submission.pkl"
        save_path = save_dir / save_name
        with open(save_path, "wb") as f:
            pickle.dump(scene, f)

        print(
            f"[✅ Saved] {save_path} | num_valid={num_valid_nodes}, "
            f"categories(unique)={np.unique(categories)}"
        )


if __name__ == "__main__":
    parquet_to_pkl_with_dataset(
        parquet_path="submission/single_agent_2025-10-05-17-33.parquet",
        dataset_root="data/DeMo_ETRI_processed/test/", 
        save_dir="submission_pkl/DeMo_first_submission/",
        use_pt=True,
    )
