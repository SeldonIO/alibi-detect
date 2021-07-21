import numpy as np
import tensorflow as tf
from typing import Tuple, Union


class TFDataset(tf.keras.utils.Sequence):
    def __init__(self, x: Union[np.ndarray, list], y: np.ndarray, batch_size: int, shuffle: bool = True) -> None:
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __getitem__(self, idx: int) -> Tuple[Union[np.ndarray, tuple], np.ndarray]:
        istart, istop = idx * self.batch_size, (idx + 1) * self.batch_size
        x = self.x[istart:istop]
        y = self.y[istart:istop]
        return x, y

    def __len__(self) -> int:
        return len(self.x) // self.batch_size

    def on_epoch_end(self) -> None:
        if self.shuffle:
            perm = np.random.permutation(len(self.x))
            self.x = [self.x[i] for i in perm] if isinstance(self.x, list) else self.x[perm]
            self.y = self.y[perm]
