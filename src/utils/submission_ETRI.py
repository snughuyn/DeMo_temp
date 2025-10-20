import os
import time
import torch
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from torch import Tensor
from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission
import multiprocessing as mp


class SubmissionETRI:
    def __init__(self, save_dir: str = "DeMo/submission", dataset_root: str = "data/DeMo_ETRI_processed_lane/test/") -> None:
        stamp = time.strftime("%Y-%m-%d-%H-%M", time.localtime())

        # 🔹 시간 폴더 생성
        self.base_dir = Path(save_dir) / stamp
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # 🔹 parquet 파일과 pkl 폴더 경로 설정
        self.submission_file = self.base_dir / f"{stamp}.parquet"
        self.pkl_save_dir = self.base_dir / "submission"
        self.pkl_save_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_root = Path(dataset_root)
        self.challenge_submission = ChallengeSubmission(predictions={})

    def format_data(
        self,
        data: dict,
        trajectory: Tensor,
        probability: Tensor,
        normalized_probability=False,
        inference=False,
    ) -> None:
        scenario_ids = data["scenario_id"]
        track_ids = data["track_id"]
        batch = len(track_ids)

        origin = data["origin"].view(batch, 1, 1, 2).double()
        theta = data["theta"].double()
        rotate_mat = torch.stack(
            [
                torch.cos(theta),
                torch.sin(theta),
                -torch.sin(theta),
                torch.cos(theta),
            ],
            dim=1,
        ).reshape(batch, 2, 2)

        with torch.no_grad():
            global_trajectory = (
                torch.matmul(trajectory[..., :2].double(), rotate_mat.unsqueeze(1))
                + origin
            )
            if not normalized_probability:
                probability = torch.softmax(probability.double(), dim=-1)

        global_trajectory = global_trajectory.detach().cpu().numpy()
        probability = probability.detach().cpu().numpy()

        if inference:
            return global_trajectory, probability

        for i, (scene_id, track_id) in enumerate(zip(scenario_ids, track_ids)):
            if scene_id not in self.challenge_submission.predictions:
                self.challenge_submission.predictions[scene_id] = {}

            self.challenge_submission.predictions[scene_id][track_id] = (
                global_trajectory[i],
                probability[i],
            )

    def generate_submission_file(self):
        print("generating submission file for ETRI trajectory prediction challenge")
        self.challenge_submission.to_parquet(self.submission_file)
        print(f"[✅] parquet saved to {self.submission_file}")

        try:
            df = pd.read_parquet(self.submission_file)
            args_list = [
                (scenario_id, scene_df, self.dataset_root, self.pkl_save_dir)
                for scenario_id, scene_df in df.groupby("scenario_id")
            ]

            print(f"[🚀] Converting {len(args_list)} scenes")

            with mp.Pool(processes=min(16, mp.cpu_count())) as pool:
                results = list(pool.imap_unordered(_convert_single_scene, args_list))

            # ✅ 변환 성공/실패 요약
            success = sum(1 for r in results if r and r.startswith("[✅"))
            failed = sum(1 for r in results if r and r.startswith("[❌"))

            print(f"[🏁] Conversion complete! ✅ {success} succeeded | ❌ {failed} failed")
            print(f"[📁] All PKL files saved under: {self.pkl_save_dir}/")

        except Exception as e:
            print(f"[❌] Failed to convert parquet to PKL: {e}")


def _convert_single_scene(args):
    """하나의 scenario_id를 pkl로 변환 (병렬용 함수)"""
    scenario_id, scene_df, dataset_root, save_dir = args
    try:
        file_name = f"log_{scenario_id}.pt"
        file_path = Path(dataset_root) / file_name
        if not file_path.exists():
            return f"[❌] Missing scene file: {file_name}"

        scene_data = torch.load(file_path)
        agent_ids = scene_data.get("agent_ids", [])
        x_attr = scene_data.get("x_attr", None)
        scored_idx = scene_data.get("scored_idx", [])
        num_nodes = len(agent_ids)
        num_valid_nodes = len(scored_idx) + 1

        if isinstance(x_attr, torch.Tensor):
            x_attr = x_attr.cpu().numpy()
        elif isinstance(x_attr, list):
            x_attr = np.array(x_attr)
        if x_attr is not None and x_attr.ndim == 2 and x_attr.shape[1] >= 2:
            categories = x_attr[:, 1].astype(int)
        else:
            categories = np.zeros(num_nodes, dtype=int)

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

        clean_id = scenario_id.replace("_masked", "")
        parts = clean_id.split("_")
        log_id, frm_idx_str = parts[0], parts[1] if len(parts) > 1 else "0"

        scene = {
            "log_id": log_id,
            "frm_idx": int(frm_idx_str),
            "agent": {
                "num_nodes": num_nodes,
                "num_valid_nodes": num_valid_nodes,
                "id": agent_ids,
                "category": np.array(categories, dtype=int),
                "predictions": predictions,
            },
        }

        save_path = Path(save_dir) / f"log_{log_id}_{frm_idx_str}_submission.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(scene, f)

        return "[✅ Saved]"

    except Exception as e:
        return f"[❌] Error: {e}"

