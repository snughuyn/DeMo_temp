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
    def __init__(self, filepath, patience, mode, monitor="new_minADE_avg"): # 👈 인자 추가
        super().__init__() 
        self.filepath = filepath
        # EarlyStopping 인스턴스를 내부에서 생성하여 제어
        self.early_stop = EarlyStopping(
            monitor=monitor, patience=patience, mode=mode, verbose=True
        )
        # 파일 처음 만들 때 헤더 작성
        with open(self.filepath, "w") as f:
            f.write("epoch,val_minADE1,val_minADE6,val_new_minADE1,val_new_minADE6,new_minADE_avg\n")

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        # ... (val_minADE1, val_minADE6, val_new_minADE1, val_new_minADE6 추출 및 new_minADE_avg 계산 로직 유지) ...
        val_minADE1 = metrics.get("val_minADE1")
        val_minADE6 = metrics.get("val_minADE6")
        val_new_minADE1 = metrics.get("val_new_minADE1")
        val_new_minADE6 = metrics.get("val_new_minADE6")

        if val_new_minADE1 is not None and val_new_minADE6 is not None:
            new_minADE_avg = (val_new_minADE1 + val_new_minADE6) / 2
        else:
            new_minADE_avg = None

        if new_minADE_avg is not None:
            # 1. new_minADE_avg를 강제로 metrics에 추가
            trainer.callback_metrics["new_minADE_avg"] = new_minADE_avg

        # 2. 파일 쓰기 로직 (기존과 동일)
        if not os.path.exists(self.filepath) or os.path.getsize(self.filepath) == 0:
            with open(self.filepath, "a") as f:
                f.write("epoch,val_minADE1,val_minADE6,val_new_minADE1,val_new_minADE6,new_minADE_avg\n")

        if all(v is not None for v in [val_minADE1, val_minADE6, val_new_minADE1, val_new_minADE6]) and new_minADE_avg is not None:
            with open(self.filepath, "a") as f:
                f.write(f"{epoch},{val_minADE1:.6f},{val_minADE6:.6f},{val_new_minADE1:.6f},{val_new_minADE6:.6f},{new_minADE_avg:.6f}\n")
        
        # 3. 계산 직후 EarlyStopping 체크 로직 강제 호출
        # 이 시점에는 new_minADE_avg가 확실하게 metrics에 존재합니다.
        self.early_stop.on_validation_epoch_end(trainer, pl_module)

    # EarlyStopping의 on_train_epoch_end도 강제 호출해야 합니다.
    def on_train_epoch_end(self, trainer, pl_module):
        self.early_stop.on_train_epoch_end(trainer, pl_module)


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _save_checkpoint(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        # --- [안전 장치 추가] new_minADE_avg가 없으면 직접 계산하여 삽입 ---
        
        if "new_minADE_avg" not in metrics:
            val_new_minADE1 = metrics.get("val_new_minADE1")
            val_new_minADE6 = metrics.get("val_new_minADE6")
            
            if val_new_minADE1 is not None and val_new_minADE6 is not None:
                # 텐서 또는 스칼라 값 처리
                try:
                    val_new_minADE1_f = val_new_minADE1.item()
                    val_new_minADE6_f = val_new_minADE6.item()
                except AttributeError:
                    val_new_minADE1_f = val_new_minADE1
                    val_new_minADE6_f = val_new_minADE6
                    
                new_minADE_avg = (val_new_minADE1_f + val_new_minADE6_f) / 2
                
                # 강제로 metrics에 추가하여 EarlyStopping과 Checkpoint가 찾도록 함
                metrics["new_minADE_avg"] = new_minADE_avg
        # -------------------------------------------------------------------
        
        # 기존 로직: metrics에 new_minADE_avg가 있다면 모니터를 설정합니다.
        if "new_minADE_avg" in metrics:
            avg_metric = metrics["new_minADE_avg"]
            if avg_metric is not None:
                self.monitor = "new_minADE_avg"  # 모니터링할 metric 설정
        
        # `_save_checkpoint()`에 필요한 인자 전달
        # 이제 metrics에 new_minADE_avg가 포함되어 있거나, 포함되지 않아 monitor가 변경되지 않았거나 둘 중 하나입니다.
        super()._save_checkpoint(trainer, pl_module)

class SafeEarlyStopping(EarlyStopping):
    """
    EarlyStopping을 상속받아, new_minADE_avg와 같은 동적 생성 지표가
    콜백 타이밍 문제로 누락될 경우를 대비해 강제 계산/주입 로직을 추가합니다.
    """
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        
        # EarlyStopping이 모니터링하는 지표가 현재 metrics에 없을 경우 안전 장치 발동
        if self.monitor not in metrics and self.monitor == "new_minADE_avg":
            val_new_minADE1 = metrics.get("val_new_minADE1")
            val_new_minADE6 = metrics.get("val_new_minADE6")
            
            # 기본 지표가 존재하는지 확인 (지표가 있다면 평균을 계산하여 강제 삽입)
            if val_new_minADE1 is not None and val_new_minADE6 is not None:
                try:
                    # 텐서인 경우 item()으로 스칼라 값을 추출
                    val_new_minADE1_f = val_new_minADE1.item()
                    val_new_minADE6_f = val_new_minADE6.item()
                except AttributeError:
                    # 이미 스칼라 값인 경우 그대로 사용
                    val_new_minADE1_f = val_new_minADE1
                    val_new_minADE6_f = val_new_minADE6
                    
                new_minADE_avg = (val_new_minADE1_f + val_new_minADE6_f) / 2
                metrics["new_minADE_avg"] = new_minADE_avg
        
        # 부모 클래스의 Early Stopping 체크 로직 실행 (이제 지표가 존재함)
        super().on_train_epoch_end(trainer, pl_module)

@hydra.main(version_base=None, config_path="conf", config_name="config_ETRI_lane")
def main(conf):
    pl.seed_everything(conf.seed, workers=True)
    output_dir = HydraConfig.get().runtime.output_dir

    logger = TensorBoardLogger(save_dir=output_dir, name="logs")

    # ✅ CSV 저장 Callback 추가
    metrics_csv_path = os.path.join(output_dir, "val_metrics.csv")
    save_metrics_callback = SaveSelectedMetricsCallback(
        filepath=metrics_csv_path,
        patience=15,          # 👈 EarlyStopping 파라미터
        mode='min',           # 👈 EarlyStopping 파라미터
        monitor='new_minADE_avg' # 👈 EarlyStopping 파라미터
    )

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
        use_distributed_sampler=False,
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
