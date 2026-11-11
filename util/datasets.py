"""Small adapters around torchvision's official CelebA dataset."""
from collections.abc import Callable
from pathlib import Path

from torch.utils.data import Dataset
from torchvision.datasets import CelebA


def constant_label(_target: object) -> int:
    return 0


def celeba_train(root: str | Path, transform: Callable) -> CelebA:
    return CelebA(root, split='train', target_type='identity', transform=transform,
                  target_transform=constant_label, download=False)


class ImageOnlyDataset(Dataset):
    """The standard FID library expects image tensors without target tuples."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index][0]
