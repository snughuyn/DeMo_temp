# import os
# import sys
# import hydra
# import pytorch_lightning as pl
# from hydra.core.hydra_config import HydraConfig
# from hydra.utils import instantiate
# from pytorch_lightning.callbacks import (
#     LearningRateMonitor,
#     ModelCheckpoint,
#     RichModelSummary,
#     RichProgressBar,
#     Callback,
#     EarlyStopping
# )
# from pytorch_lightning.loggers import TensorBoardLogger



# class SaveSelectedMetricsCallback(Callback):
#     def __init__(self, filepath):
#         super().__init__() 
#         self.filepath = filepath
#         # 파일 처음 만들 때 헤더 작성
#         with open(self.filepath, "w") as f:
#             f.write("epoch,val_minADE1,val_minADE6,val_new_minADE1,val_new_minADE6\n")

#     def on_validation_epoch_end(self, trainer, pl_module):
#         metrics = trainer.callback_metrics
#         epoch = trainer.current_epoch
#         val_minADE1 = metrics.get("val_minADE1")
#         val_minADE6 = metrics.get("val_minADE6")
#         val_new_minADE1 = metrics.get("val_new_minADE1")  #added
#         val_new_minADE6 = metrics.get("val_new_minADE6")  #added

#         if not os.path.exists(self.filepath) or os.path.getsize(self.filepath) == 0:
#             with open(self.filepath, "a") as f:
#                 f.write("epoch,val_minADE1,val_minADE6,val_new_minADE1,val_new_minADE6\n")

#         # 값이 존재할 때만 기록
#         if all(v is not None for v in [val_minADE1, val_minADE6, val_new_minADE1, val_new_minADE6]):
#             with open(self.filepath, "a") as f:
#                 f.write(f"{epoch},{val_minADE1:.6f},{val_minADE6:.6f},{val_new_minADE1:.6f},{val_new_minADE6:.6f}\n")

#         # # 값이 있을 때만 기록
#         # if val_minADE1 is not None and val_minADE6 is not None:
#         #     with open(self.filepath, "a") as f:
#         #         f.write(f"{epoch},{val_minADE1:.6f},{val_minADE6:.6f}\n")


# @hydra.main(version_base=None, config_path="conf", config_name="config_ETRI_lane")
# def main(conf):
#     pl.seed_everything(conf.seed, workers=True)
#     output_dir = HydraConfig.get().runtime.output_dir

#     logger = TensorBoardLogger(save_dir=output_dir, name="logs")

#     # ✅ CSV 저장 Callback 추가
#     metrics_csv_path = os.path.join(output_dir, "val_metrics.csv")
#     save_metrics_callback = SaveSelectedMetricsCallback(metrics_csv_path)

#     callbacks = [
#         ModelCheckpoint(
#             dirpath=os.path.join(output_dir, "checkpoints"),
#             filename="{epoch}",
#             monitor=f"{conf.monitor}",
#             mode="min",
#             save_top_k=conf.save_top_k,
#             save_last=True,
#         ),
#         RichModelSummary(max_depth=1),
#         RichProgressBar(),
#         LearningRateMonitor(logging_interval="epoch"),
#         save_metrics_callback,
#         EarlyStopping(
#             monitor='val_new_minADE1',
#             patience=20,
#             mode='min',
#             verbose=True
#         )
#     ]

#     trainer = pl.Trainer(
#         logger=logger,
#         gradient_clip_val=conf.gradient_clip_val,
#         gradient_clip_algorithm=conf.gradient_clip_algorithm,
#         max_epochs=conf.epochs,
#         accelerator="gpu",
#         devices=conf.gpus,
#         strategy="ddp_find_unused_parameters_true",
#         callbacks=callbacks,
#         limit_train_batches=conf.limit_train_batches,
#         limit_val_batches=conf.limit_val_batches,
#         sync_batchnorm=conf.sync_bn,
    
#     )

#     model = instantiate(conf.model.target)
#     os.system('cp -a %s %s' % ('conf', output_dir))
#     os.system('cp -a %s %s' % ('src', output_dir))
#     with open(f'{output_dir}/model.txt', 'w') as f:
#         original_stdout = sys.stdout
#         sys.stdout = f
#         print(model)
#         sys.stdout = original_stdout

#     datamodule = instantiate(conf.datamodule.target)

#     trainer.fit(model, datamodule, ckpt_path=conf.checkpoint)
#     trainer.validate(model, datamodule.val_dataloader())


# if __name__ == "__main__":
#     main()
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
    EarlyStopping
)
from pytorch_lightning.loggers import TensorBoardLogger

class SaveSelectedMetricsCallback(Callback):
    def __init__(self, filepath):
        super().__init__() 
        self.filepath = filepath
        # 파일 처음 만들 때 헤더 작성
        with open(self.filepath, "w") as f:
            f.write("epoch,val_minADE1,val_minADE6,val_new_minADE1,val_new_minADE6,new_minADE_avg\n")

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        val_minADE1 = metrics.get("val_minADE1")
        val_minADE6 = metrics.get("val_minADE6")
        val_new_minADE1 = metrics.get("val_new_minADE1")  # 추가된 부분
        val_new_minADE6 = metrics.get("val_new_minADE6")  # 추가된 부분

        # 평균값 계산 (minADE1과 minADE6의 평균)
        if val_new_minADE1 is not None and val_new_minADE6 is not None:
            new_minADE_avg = (val_new_minADE1 + val_new_minADE6) / 2
        else:
            new_minADE_avg = None

        # trainer.callback_metrics에 new_minADE_avg 추가
        if new_minADE_avg is not None:
            trainer.callback_metrics["new_minADE_avg"] = new_minADE_avg

        # 파일이 비었거나 존재하지 않으면 헤더 작성
        if not os.path.exists(self.filepath) or os.path.getsize(self.filepath) == 0:
            with open(self.filepath, "a") as f:
                f.write("epoch,val_minADE1,val_minADE6,val_new_minADE1,val_new_minADE6,new_minADE_avg\n")

        # 값이 존재할 때만 기록
        if all(v is not None for v in [val_minADE1, val_minADE6, val_new_minADE1, val_new_minADE6]) and new_minADE_avg is not None:
            with open(self.filepath, "a") as f:
                f.write(f"{epoch},{val_minADE1:.6f},{val_minADE6:.6f},{val_new_minADE1:.6f},{val_new_minADE6:.6f},{new_minADE_avg:.6f}\n")


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _save_checkpoint(self, trainer, pl_module):
        # Custom logic: Save checkpoint based on new_minADE_avg
        metrics = trainer.callback_metrics
        if "new_minADE_avg" in metrics:
            avg_metric = metrics["new_minADE_avg"]
            # Ensure it's logged as a scalar
            if avg_metric is not None:
                self.monitor = "new_minADE_avg"  # 모니터링할 metric 설정
        # `_save_checkpoint()`에 필요한 인자 전달
        super()._save_checkpoint(trainer, pl_module)  # 수정된 부분



@hydra.main(version_base=None, config_path="conf", config_name="config_ETRI_lane")
def main(conf):
    pl.seed_everything(conf.seed, workers=True)
    output_dir = HydraConfig.get().runtime.output_dir

    logger = TensorBoardLogger(save_dir=output_dir, name="logs")

    # ✅ CSV 저장 Callback 추가
    metrics_csv_path = os.path.join(output_dir, "val_metrics.csv")
    save_metrics_callback = SaveSelectedMetricsCallback(metrics_csv_path)

    # Custom ModelCheckpoint with the adjusted monitor key
    checkpoint_callback = CustomModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="{epoch}",
        monitor="new_minADE_avg",  # custom 평균 값 모니터링
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    callbacks = [
        checkpoint_callback,
        RichModelSummary(max_depth=1),
        RichProgressBar(),
        LearningRateMonitor(logging_interval="epoch"),
        save_metrics_callback,
        EarlyStopping(
            monitor='new_minADE_avg',  # custom 평균 값 모니터링
            patience=15,
            mode='min',
            verbose=True
        )
    ]

    trainer = pl.Trainer(
        logger=logger,
        gradient_clip_val=conf.gradient_clip_val,
        gradient_clip_algorithm=conf.gradient_clip_algorithm,
        max_epochs=conf.epochs,
        accelerator="gpu",
        devices=conf.gpus,
        strategy="ddp_find_unused_parameters_true",
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
