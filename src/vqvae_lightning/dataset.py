"""Simple DataModule Implementation."""

import tarfile
from collections.abc import Callable
from pathlib import Path

import gdown
import numpy as np
import torch
from lightning import LightningDataModule
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def split_train_validation(
    path_list: list[Path],
    train_ratio: float = 0.8,
) -> tuple[list[Path], list[Path]]:
    """Pathのリストを`train_ratio`で分割する."""
    split_point = int(len(path_list) * train_ratio)
    return path_list[:split_point], path_list[split_point:]


class EpisodeObservaionDataset(Dataset[tuple[Tensor, Tensor]]):
    """観測のデータセット."""

    def __init__(
        self,
        path_list: list[Path],
        augmentation: Callable[[Tensor], Tensor],
    ) -> None:
        super().__init__()
        self.path_list = path_list
        self.augmentation = augmentation

    def __len__(self) -> int:
        """Return the number of data."""
        return len(self.path_list)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return the data at the index."""
        observation = torch.load(self.path_list[idx])
        return self.augmentation(observation), observation


class EpisodeObservationDataModule(LightningDataModule):
    """行動-観測シーケンスのデータの観測事前学習用 DataModule."""

    def __init__(
        self,
        *,
        batch_size: int,
        num_workers: int,
        data_name: str,
        gdrive_url: str,
        preprocess: Callable[[Tensor], Tensor] | None = None,
        augmentation: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_name = data_name
        self.gdrive_url = gdrive_url
        self.data_dir = Path("data") / data_name
        self.processed_data_dir = self.data_dir.parent / f"{self.data_dir.stem}_processed"
        self.preprocess = preprocess or nn.Identity()
        self.augmentation = augmentation or nn.Identity()

    def prepare_data(self) -> None:
        """データの前処理・保存."""
        if not self.data_dir.exists():
            self.load_from_gdrive()

        if self.processed_data_dir.exists():
            return

        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        data_count = 0
        for path in tqdm(self.data_dir.glob("observation_*.npy")):
            for observation in np.load(path):
                save_path = self.processed_data_dir / f"observation_{data_count:06}.pt"
                processed_observation = self.preprocess(observation)
                torch.save(processed_observation.detach().clone(), save_path)
                data_count += 1

    def load_from_gdrive(self) -> None:
        """Google Drive からデータをダウンロードする."""
        filename = gdown.download(self.gdrive_url, quiet=False, fuzzy=True)
        tarfile.open(filename, "r:gz").extractall(path=Path("data"), filter="data")
        Path(filename).unlink(missing_ok=False)

    def setup(self, stage: str) -> None:  # noqa: ARG002
        """Train/Val/Testデータセットのセットアップ."""
        path_list = sorted(self.processed_data_dir.glob("observation_*.pt"))
        train_path_list, val_path_list = split_train_validation(path_list)
        self.train_dataset = EpisodeObservaionDataset(
            path_list=train_path_list,
            augmentation=self.augmentation,
        )
        self.val_dataset = EpisodeObservaionDataset(
            path_list=val_path_list,
            augmentation=nn.Identity(),
        )

    def train_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        """Return train dataloader."""
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        """Return validation dataloader."""
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
