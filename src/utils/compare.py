import os
import pickle
import numpy as np

def compare_qcnet_pkl_dirs(dir_a, dir_b):
    """
    Compare structure consistency between two QCNet-style result directories.

    Args:
        dir_a (str): baseline directory path
        dir_b (str): new model directory path
    """
    files_a = sorted([f for f in os.listdir(dir_a) if f.endswith(".pkl")])
    files_b = sorted([f for f in os.listdir(dir_b) if f.endswith(".pkl")])
    common = sorted(list(set(files_a) & set(files_b)))

    print(f"✅ Total common files: {len(common)}\n")

    mismatched = []

    for fname in common:
        path_a = os.path.join(dir_a, fname)
        path_b = os.path.join(dir_b, fname)

        with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
            data_a = pickle.load(fa)
            data_b = pickle.load(fb)

        def check_equal(a, b, name):
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                return np.array_equal(a, b)
            return a == b

        checks = {
            "log_id": check_equal(data_a["log_id"], data_b["log_id"], "log_id"),
            "frm_idx": check_equal(data_a["frm_idx"], data_b["frm_idx"], "frm_idx"),
            "num_nodes": check_equal(data_a["agent"]["num_nodes"], data_b["agent"]["num_nodes"], "num_nodes"),
            "num_valid_nodes": check_equal(data_a["agent"]["num_valid_nodes"], data_b["agent"]["num_valid_nodes"], "num_valid_nodes"),
            "id_list": data_a["agent"]["id"] == data_b["agent"]["id"],
            "category": np.array_equal(data_a["agent"]["category"], data_b["agent"]["category"]),
            "pred_shape": np.array_equal(data_a["agent"]["predictions"].shape, data_b["agent"]["predictions"].shape),
        }

        if not all(checks.values()):
            mismatched.append((fname, checks))

    if not mismatched:
        print("🎯 All matched perfectly! Structures are identical.")
    else:
        print(f"⚠️ Found {len(mismatched)} mismatched files:\n")
        for fname, info in mismatched:
            print(f"── {fname}")
            for key, ok in info.items():
                if not ok:
                    print(f"   ✗ {key} mismatch")
            print()

if __name__ == "__main__":
    compare_qcnet_pkl_dirs(
        dir_a="/home/yaaaaaaaang/DeMo/submission_pkl/DeMo_first_submission",
        dir_b="/home/yaaaaaaaang/submission/prediction_results"
    )
