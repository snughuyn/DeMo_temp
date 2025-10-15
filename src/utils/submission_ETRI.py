# src/utils/submission_ETRI.py
import os
import time
import pickle
import torch
import numpy as np
from pathlib import Path


class SubmissionETRI:
    def __init__(self, save_dir: str = "") -> None:
        # 저장 경로 생성
        stamp = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        self.save_dir = Path(save_dir) / f"submission_{stamp}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.num_saved = 0

    def format_data(
        self,
        data: dict,
        trajectory: torch.Tensor,   # (N, M, 60, 2)
        probability: torch.Tensor,  # (N, M)
        normalized_probability=False,
        inference=False,
    ) -> None:
        """
        ETRI 포맷 데이터를 QCNet baseline 스타일 .pkl로 저장
        """
        scenario_id = data["scenario_id"]          # str
        agent_ids = data["agent_ids"]              # list[str], 길이 N
        scored_idx = data.get("scored_idx", [])    # 평가대상 agent index
        num_nodes = len(agent_ids)
        num_valid_nodes = len(scored_idx) if scored_idx else num_nodes

        # === 1️⃣ local → global 좌표 변환 ===
        origin = torch.tensor(data["x_positions"])[:, -1, :2]   # (N, 2)
        theta = torch.tensor(data["x_angles"])[:, -1]           # (N,)

        rot_mat = torch.zeros(num_nodes, 2, 2)
        rot_mat[:, 0, 0] = torch.cos(theta)
        rot_mat[:, 0, 1] = torch.sin(theta)
        rot_mat[:, 1, 0] = -torch.sin(theta)
        rot_mat[:, 1, 1] = torch.cos(theta)

        global_traj = (
            torch.matmul(trajectory[..., :2], rot_mat.unsqueeze(1))
            + origin[:, None, None, :]
        )

        if not normalized_probability:
            probability = torch.softmax(probability, dim=-1)

        global_traj = global_traj.detach().cpu().numpy()
        probability = probability.detach().cpu().numpy()

        # === 2️⃣ category 생성 ===
        if "x_attr" in data:
            try:
                category = np.array(data["x_attr"])[:, 0].astype(int)
            except Exception:
                category = np.zeros(num_nodes, dtype=int)
        else:
            category = np.zeros(num_nodes, dtype=int)

        # === 3️⃣ QCNet-style scene dict ===
        scene = {
            "log_id": str(scenario_id),
            "frm_idx": int(data.get("focal_idx", 0)),  # focal frame index
            "agent": {
                "num_nodes": num_nodes,
                "num_valid_nodes": num_valid_nodes,
                "id": agent_ids,
                "category": category,
                "predictions": global_traj,   # (N, M, 60, 2)
            },
        }

        # === 4️⃣ 저장 ===
        save_path = self.save_dir / f"log_{scenario_id}_{scene['frm_idx']:07d}_submission.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(scene, f)

        self.num_saved += 1

    def generate_submission_file(self):
        print(f"[Submission] Saved {self.num_saved} scenes to {self.save_dir}")
