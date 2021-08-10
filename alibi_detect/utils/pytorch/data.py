import numpy as np
import torch
from typing import Tuple, Union


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, *indexables: Tuple[Union[np.ndarray, torch.Tensor, list], ...]) -> None:
        self.indexables = indexables

    def __getitem__(self, idx: int) -> Tuple[Union[np.ndarray, torch.Tensor, tuple], ...]:
        output = tuple(indexable[idx] for indexable in self.indexables)
        return output if len(output) > 1 else output[0]

    def __len__(self) -> int:
        return len(self.indexables[0])
