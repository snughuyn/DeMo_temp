import os
import sys
import hydra
from hydra.core.hydra_config import HydraConfig
import pytorch_lightning as pl
from hydra.utils import instantiate, to_absolute_path
from thop import profile
import torch


# ---------------------------
# Utility: move batch to device recursively
# ---------------------------
def move_to_device(obj, device, dtype=None):
    if hasattr(obj, "to"):
        try:
            return obj.to(device=device, dtype=dtype) if dtype and obj.is_floating_point() else obj.to(device)
        except Exception:
            try:
                return obj.to(device)
            except Exception:
                pass
    if isinstance(obj, dict):
        return {k: move_to_device(v, device, dtype) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        out = [move_to_device(v, device, dtype) for v in obj]
        return tuple(out) if isinstance(obj, tuple) else out
    return obj


@hydra.main(version_base=None, config_path="./conf/", config_name="config_ETRI_test")
def main(conf):
    # ---------------------------
    # Setup
    # ---------------------------
    pl.seed_everything(conf.seed)
    output_dir = HydraConfig.get().runtime.output_dir
    checkpoint = to_absolute_path("/home/tako/DeMo_6/outputs/change_av2_ETRI_lane_default_baseline_default/20251017-123321/checkpoints/epoch=20.ckpt")
    assert os.path.exists(checkpoint), f"Checkpoint {checkpoint} does not exist"

    trainer = pl.Trainer(
        logger=False,
        accelerator="gpu",
        devices=conf.gpus,
        max_epochs=1,
        limit_val_batches=conf.limit_val_batches,
        limit_test_batches=conf.limit_test_batches,
    )

    datamodule: pl.LightningDataModule = instantiate(conf.datamodule.target, test=conf.test)
    model = instantiate(conf.model.target)
    os.system(f'cp -a conf {output_dir}')
    os.system(f'cp -a src {output_dir}')

    # ---------------------------
    # Load Checkpoint
    # ---------------------------
    print("\n🚀 Running model evaluation...\n")
    trainer.test(model, datamodule, ckpt_path=checkpoint)

    # ---------------------------
    # FLOPs Measurement (100 samples)
    # ---------------------------
    try:
        print("\n🔍 Measuring average FLOPs over 100 samples...\n")

        test_loader = datamodule.test_dataloader()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        torch.set_float32_matmul_precision("medium")

        total_gflops = 0.0
        count = 0

        for i, batch in enumerate(test_loader):
            if i >= 100:  # only 100 samples
                break
            try:
                model_dtype = next(model.parameters()).dtype
                batch = move_to_device(batch, device, dtype=model_dtype)

                macs, params = profile(model, inputs=(batch,), verbose=False)
                total_gflops += 2 * macs / 1e9
                count += 1

                if (i + 1) % 10 == 0:
                    print(f"  → Processed {i + 1}/100 samples")

            except Exception as e:
                print(f"[Skip] Sample #{i} skipped due to error: {e}")
                continue

        if count == 0:
            print("\n[Warning] No valid samples processed for FLOPs calculation.")
        else:
            avg_gflops = total_gflops / count
            print(f"\nThe estimated FLOPs : {avg_gflops:.2f} G")

            # ✅ 한 줄짜리 결과 파일 저장
            with open(os.path.join(output_dir, "flops_result.txt"), "w") as f:
                f.write(f"The estimated FLOPs : {avg_gflops:.2f} G\n")

    except Exception as e:
        print(f"[Warning] FLOPs 계산 중 오류 발생: {e}")


if __name__ == "__main__":
    main()
