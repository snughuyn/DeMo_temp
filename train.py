import os
import sys
import hydra
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    Callback,
)
from pytorch_lightning.loggers import TensorBoardLogger
import csv


# ✅ 커스텀 콜백: CSV 저장 + top5 관리
class CSVMetricLogger(Callback):
    def __init__(self, save_dir):
        super().__init__()
        self.save_path = os.path.join(save_dir, "metrics.csv")
        self.best_epochs = []  # (epoch, mean) 저장
        # 헤더 작성
        with open(self.save_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "val_minADE1", "val_minADE6", "val_minADE_mean"])

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        val_minADE1 = metrics.get("val_minADE1")
        val_minADE6 = metrics.get("val_minADE6")

        if val_minADE1 is not None and val_minADE6 is not None:
            val_minADE1 = float(val_minADE1)
            val_minADE6 = float(val_minADE6)
            val_mean = 0.5 * (val_minADE1 + val_minADE6)

            # CSV에 기록
            with open(self.save_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([trainer.current_epoch, val_minADE1, val_minADE6, val_mean])

            # ✅ top5 관리
            self.best_epochs.append((trainer.current_epoch, val_mean))
            self.best_epochs.sort(key=lambda x: x[1])  # 작은 값이 좋은 성능
            self.best_epochs = self.best_epochs[:5]    # top5 유지

            # ✅ top5 체크포인트만 복사
            ckpt_dir = os.path.join(trainer.logger.save_dir, "checkpoints_top5_mean")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, f"{trainer.current_epoch}.ckpt")
            if os.path.exists(ckpt_path):
                target_path = os.path.join(ckpt_dir, f"epoch{trainer.current_epoch}-mean{val_mean:.4f}.ckpt")
                os.system(f"cp {ckpt_path} {target_path}")


@hydra.main(version_base=None, config_path="conf", config_name="config_ETRI_width")
def main(conf):
    pl.seed_everything(conf.seed, workers=True)
    output_dir = HydraConfig.get().runtime.output_dir

    logger = TensorBoardLogger(save_dir=output_dir, name="logs")

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename="{epoch}",
            monitor=f"{conf.monitor}",
            mode="min",
            save_top_k=conf.save_top_k,
            save_last=True,
        ),
        RichModelSummary(max_depth=1),
        RichProgressBar(),
        LearningRateMonitor(logging_interval="epoch"),
        CSVMetricLogger(output_dir),   # ✅ CSV + top5 관리
    ]

    trainer = pl.Trainer(
        logger=logger,
        gradient_clip_val=conf.gradient_clip_val,
        gradient_clip_algorithm=conf.gradient_clip_algorithm,
        max_epochs=conf.epochs,
        accelerator="gpu",
        devices=conf.gpus,
        strategy="ddp_find_unused_parameters_false",
        callbacks=callbacks,
        limit_train_batches=conf.limit_train_batches,
        limit_val_batches=conf.limit_val_batches,
        sync_batchnorm=conf.sync_bn,
    )

    model = instantiate(conf.model.target)
    os.system('cp -a %s %s' % ('conf', output_dir))
    os.system('cp -a %s %s' % ('src', output_dir))
    with open(f'{output_dir}/model.txt', 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        print(model)
        sys.stdout = original_stdout

    datamodule = instantiate(conf.datamodule.target)
    trainer.fit(model, datamodule, ckpt_path=conf.checkpoint)
    trainer.validate(model, datamodule.val_dataloader())


if __name__ == "__main__":
    main()
