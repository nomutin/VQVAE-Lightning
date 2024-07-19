"""Simple DataModule Implementation."""

from collections.abc import Callable
from pathlib import Path

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


class MixDataModule(LightningDataModule):
    """DataModule for CIFAR100(Train/Val) & MNIST(Test) dataset."""

    def __init__(self, root: str, batch_size: int, transform: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.transform = transform

    def prepare_data(self) -> None:
        """Download datasets."""
        from torchvision.datasets import CIFAR100, Food101  # noqa: PLC0415

        CIFAR100(self.root, train=True, download=True)
        CIFAR100(self.root, train=False, download=True)
        Food101(self.root, split="train", download=True)
        Food101(self.root, split="test", download=True)

    def setup(self, stage: str) -> None:
        """Set up train/val/test dataset."""
        if stage == "fit" or stage is None:
            from torchvision.datasets import CIFAR100, Food101  # noqa: PLC0415

            self.train_dataset = Food101(self.root, split="train", transform=self.transform)
            self.val_dataset = Food101(self.root, split="test", transform=self.transform)
            self.test_dataset = CIFAR100(self.root, train=True, transform=self.transform)

    def train_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        """Return train dataloader."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        """Return validation dataloader."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        """Return test dataloader."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


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
    """
    行動-観測シーケンスのデータの観測事前学習用 DataModule.

    Parameters
    ----------
    data_name: str
        データセット名.
        data/{data_name}ディレクトリにデータが格納されていることを想定.
    preprocess : Callable[[Tensor], Tensor]
        画像の前処理関数.
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        data_name: str,
        preprocess: Callable[[Tensor], Tensor] | None = None,
        augmentation: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_name = data_name
        self.data_dir = Path("data") / data_name
        self.processed_data_dir = self.data_dir.parent / f"{self.data_dir.stem}_processed"
        self.preprocess = preprocess or nn.Identity()
        self.augmentation = augmentation or nn.Identity()

    def prepare_data(self) -> None:
        """データの前処理・保存."""
        if self.processed_data_dir.exists():
            return
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        data_count = 0
        for path in tqdm(self.data_dir.glob("observation_*.npy")):
            sequence = []
            for observation in np.load(path):
                save_path = self.processed_data_dir / f"observation_{data_count:06}.pt"
                processed_observation = self.preprocess(observation)
                torch.save(processed_observation.detach().clone(), save_path)
                sequence.append(processed_observation)
                data_count += 1
            save_path = self.processed_data_dir / f"sequence_{data_count:06}.pt"
            torch.save(torch.stack(sequence).detach().clone(), save_path)

    def setup(self, stage: str) -> None:
        """Train/Val/Testデータセットのセットアップ."""
        path_list = sorted(self.processed_data_dir.glob("observation_*.pt"))
        train_path_list, val_path_list = split_train_validation(path_list)
        self.train_dataset = EpisodeObservaionDataset(
            path_list=train_path_list,
            augmentation=self.augmentation,
        )
        self.val_dataset = EpisodeObservaionDataset(
            path_list=val_path_list,
            augmentation=lambda x: x,
        )
        if stage == "test":
            self.test_dataset = EpisodeObservaionDataset(
                path_list=sorted(self.processed_data_dir.glob("sequence_*.pt")),
                augmentation=lambda x: x,
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

    def test_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        """Return test dataloader."""
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
