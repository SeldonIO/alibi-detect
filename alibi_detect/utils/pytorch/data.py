import numpy as np
import torch
from typing import Tuple, Union

Indexable = Union[np.ndarray, torch.Tensor, list]


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, *indexables: Tuple[Indexable, ...]) -> None:
        self.indexables = indexables

    def __getitem__(self, idx: int) -> Union[Tuple[Indexable, ...], Indexable]:
        output = tuple(indexable[idx] for indexable in self.indexables)
        return output if len(output) > 1 else output[0]

    def __len__(self) -> int:
        return len(self.indexables[0])
