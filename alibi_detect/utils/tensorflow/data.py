import numpy as np
import tensorflow as tf
from typing import Tuple, Union

Indexable = Union[np.ndarray, tf.Tensor, list]


class TFDataset(tf.keras.utils.Sequence):
    def __init__(
        self, *indexables: Tuple[Indexable, ...], batch_size: int = int(1e10), shuffle: bool = True,
    ) -> None:
        self.indexables = indexables
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __getitem__(self, idx: int) -> Union[Tuple[Indexable, ...], Indexable]:
        istart, istop = idx * self.batch_size, (idx + 1) * self.batch_size
        output = tuple(indexable[istart:istop] for indexable in self.indexables)
        return output if len(output) > 1 else output[0]

    def __len__(self) -> int:
        return len(self.indexables[0]) // self.batch_size

    def on_epoch_end(self) -> None:
        if self.shuffle:
            perm = np.random.permutation(len(self.indexables[0]))
            self.indexables = tuple(
                [indexable[i] for i in perm] if isinstance(indexable, list) else indexable[perm]
                for indexable in self.indexables
            )
