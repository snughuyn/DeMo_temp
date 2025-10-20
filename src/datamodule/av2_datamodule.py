from pathlib import Path
from typing import Optional, Iterator, List, Tuple
import torch
import random
from torch.utils.data import DataLoader as TorchDataLoader, Sampler
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from .av2_dataset import Av2Dataset, collate_fn, get_frame_id
from pytorch_lightning import LightningDataModule
import math
import torch.distributed as dist

# ✅ 추가: 분산 샘플러 임포트
from torch.utils.data.distributed import DistributedSampler

import math
import torch
import torch.distributed as dist
from torch.utils.data import Sampler, BatchSampler

class DistributedPairBatchSampler(BatchSampler):
    """
    (t, t+1) '쌍' 단위로 셔플/샤딩/배치 구성.
    - 각 rank는 서로 다른 '쌍' 서브셋만 소비
    - 배치: pair_per_batch 쌍을 이어붙인 인덱스 [i_t, i_t+1, i_t, i_t+1, ...]
    - drop_last=True이면 모든 rank의 배치 수가 동일 → DDP에서 안전
    """
    def __init__(self, dataset, pairs, batch_size, sampler=None, shuffle=True, drop_last=True):  # ✅ sampler 인자 유지
        assert batch_size % 2 == 0, "batch_size는 2의 배수여야 (pair 단위) 안전합니다."
        self.dataset = dataset
        self.pairs = list(pairs)            # [(idx_t, idx_t1), ...]
        self.batch_size = batch_size
        self.pairs_per_batch = batch_size // 2
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.sampler = sampler  # ✅ Lightning이 주입하려 해도 에러 안 나게 홀딩

        # DDP 정보
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        self.epoch = 0
        self._recompute_shards()

    def _recompute_shards(self):
        # 1) 쌍 단위 셔플
        g = torch.Generator()
        g.manual_seed(self.epoch)
        order = torch.randperm(len(self.pairs), generator=g).tolist() if self.shuffle else list(range(len(self.pairs)))
        pairs_shuffled = [self.pairs[i] for i in order]

        # 2) world_size로 균등 샤딩 (쌍 단위)
        if self.drop_last:
            total_pairs = (len(pairs_shuffled) // self.world_size) * self.world_size
            pairs_shuffled = pairs_shuffled[:total_pairs]
        else:
            total_pairs = math.ceil(len(pairs_shuffled) / self.world_size) * self.world_size
            if len(pairs_shuffled) < total_pairs:
                pairs_shuffled += pairs_shuffled[:(total_pairs - len(pairs_shuffled))]

        self.local_pairs = pairs_shuffled[self.rank:total_pairs:self.world_size]

        # 3) 배치 수 계산
        if self.drop_last:
            self.num_batches = len(self.local_pairs) // self.pairs_per_batch
        else:
            self.num_batches = math.ceil(len(self.local_pairs) / self.pairs_per_batch)

    def __iter__(self):
        for b in range(self.num_batches):
            start = b * self.pairs_per_batch
            end   = start + self.pairs_per_batch
            cur_pairs = self.local_pairs[start:end]
            if len(cur_pairs) < self.pairs_per_batch and self.drop_last:
                break
            batch_indices = []
            for (i_t, i_t1) in cur_pairs:
                batch_indices.extend([i_t, i_t1])
            yield batch_indices

    def __len__(self):
        return self.num_batches

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self._recompute_shards()


class Av2DataModule(LightningDataModule):
    def __init__(
        self,
        data_root: str,
        dataset: dict = {},
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        test_batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 8,
        pin_memory: bool = True,
        test: bool = False,
    ):
        super(Av2DataModule, self).__init__()
        self.data_root = Path(data_root)
        self.dataset_cfg = dataset
        self.batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.test = test

        # ✅ Trainer에서 set_epoch 호출할 수 있게 보관
        self.pair_batch_sampler = None

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.test:
            self.train_dataset = Av2Dataset(data_root=self.data_root, split="train", **self.dataset_cfg)
            self.val_dataset   = Av2Dataset(data_root=self.data_root, split="val", **self.dataset_cfg)
        else:
            self.test_dataset  = Av2Dataset(data_root=self.data_root, split="test", **self.dataset_cfg)

    def _build_pairs(self):
        pairs = []
        files = self.train_dataset.file_list
        i = 0
        while i < len(files) - 1:
            scenario_id_i = files[i].stem.split('_')[1]
            scenario_id_j = files[i+1].stem.split('_')[1]
            frame_id_i = get_frame_id(files[i])
            frame_id_j = get_frame_id(files[i+1])
            if scenario_id_i == scenario_id_j and frame_id_j == frame_id_i + 1:
                pairs.append((i, i + 1))
                i += 2
            else:
                i += 1
        return pairs

    def train_dataloader(self):
        # === Main loader: 전체 17008 씬 학습 + DDP 샤딩 ===
        main_sampler = None
        if dist.is_available() and dist.is_initialized():
            main_sampler = DistributedSampler(self.train_dataset, shuffle=True, drop_last=True)  # ✅ 명시 샘플러

        main_loader = TorchDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=(main_sampler is None),    # ✅ sampler 있으면 False
            sampler=main_sampler,              # ✅ DDP 균등 샤딩
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=False,          # ✅ 데드락 방지
            prefetch_factor=2,
            drop_last=True,
            collate_fn=collate_fn,
        )

        # === Temporal loader: 쌍 단위 배치 샘플러 (DDP 샤딩 내장) ===
        pairs = self._build_pairs()
        self.pair_batch_sampler = DistributedPairBatchSampler(  # ✅ Trainer에서 set_epoch 호출
            dataset=self.train_dataset,
            pairs=pairs,
            batch_size=self.batch_size,     # 2의 배수
            shuffle=True,
            drop_last=True,
            sampler=None,                   # 자리만 채움(PL 주입 대비)
        )
        temporal_loader = TorchDataLoader(
            self.train_dataset,
            batch_sampler=self.pair_batch_sampler,  # ✅ sampler 대신 batch_sampler
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=False,
            prefetch_factor=2,
            collate_fn=collate_fn,
        )

        loaders = {"main": main_loader, "temporal": temporal_loader}
        return CombinedLoader(loaders, mode='max_size')  # ✅ 에폭 길이 = main 기준 (전 씬 보장)

    def val_dataloader(self):
        return TorchDataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=False,
            prefetch_factor=2,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return TorchDataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=False,
            prefetch_factor=2,
            collate_fn=collate_fn,
        )
