import numpy as np
import torch
from typing import Tuple, Union


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, x: Union[np.ndarray, list], y: np.ndarray) -> None:
        self.x = x
        self.y = y

    def __getitem__(self, idx: int) -> Tuple[Union[np.ndarray, tuple], np.ndarray]:
        x = self.x[idx]
        y = self.y[idx]
        return x, y

    def __len__(self) -> int:
        return len(self.x)
