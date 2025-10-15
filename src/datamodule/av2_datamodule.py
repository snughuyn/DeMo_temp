from pathlib import Path
from typing import Optional, Iterator, List, Tuple
import torch
import random
from torch.utils.data import DataLoader as TorchDataLoader, Sampler
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from .av2_dataset import Av2Dataset, collate_fn, get_frame_id
from pytorch_lightning import LightningDataModule

class PairedOnlySampler(Sampler[int]):
    def __init__(self, data_source: Av2Dataset):
        self.data_source = data_source
        self.pairs = self._create_pairs()

    def _create_pairs(self) -> List[Tuple[int, int]]:
        # ... (이 함수의 코드는 이전과 동일하게 유지)
        pairs = []
        files = self.data_source.file_list
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
        
        print(f"[PairedOnlySampler] Found {len(pairs)} pairs for temporal consistency.")
        return pairs

    def __iter__(self) -> Iterator[int]:
        random.shuffle(self.pairs)
        indices = [idx for pair in self.pairs for idx in pair]
        return iter(indices)

    def __len__(self) -> int:
        return len(self.pairs) * 2

    def __getitem__(self, index: int):
        data = torch.load(self.file_list[index])
        processed_data_list = self.process(data)
        return processed_data_list[0]

    
class Av2DataModule(LightningDataModule):
    def __init__(
        self,
        data_root: str,
        dataset: dict = {},
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        test_batch_size: int = 32,
        shuffle: bool = True, # 기존 파라미터 유지
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
        self.shuffle = shuffle # 기존 파라미터 유지
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.test = test

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.test:
            self.train_dataset = Av2Dataset(data_root=self.data_root, split="train", **self.dataset_cfg)
            self.val_dataset = Av2Dataset(data_root=self.data_root, split="val", **self.dataset_cfg)
        else:
            self.test_dataset = Av2Dataset(data_root=self.data_root, split="test", **self.dataset_cfg)

    def train_dataloader(self):
        # 1. Main Loader: 전체 데이터를 섞어서 사용
        main_loader = TorchDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

        # 2. Temporal Loader: 쌍 데이터만 사용
        temporal_sampler = PairedOnlySampler(self.train_dataset)
        temporal_loader = TorchDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=temporal_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

        loaders = {
            "main": main_loader,
            "temporal": temporal_loader
        }
        return CombinedLoader(loaders, mode='max_size_cycle')

    def val_dataloader(self):
        # Validation: 기존 로직과 동일 (순차 로드)
        return TorchDataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        # Test: 기존 로직과 동일 (순차 로드)
        return TorchDataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )