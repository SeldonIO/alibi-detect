from typing import Callable
from functools import partial
import tensorflow as tf


def activate_train_mode_for_all_layers(model: Callable) -> Callable:
    model.trainable = False  # type: ignore
    model = partial(model, training=True)  # Note this affects batchnorm etc also
    return model


def zero_diag(mat: tf.Tensor) -> tf.Tensor:
    return mat - tf.linalg.diag(tf.linalg.diag_part(mat))


def quantile(sample: tf.Tensor, p: float, type: int = 7, sorted: bool = False) -> tf.Tensor:
    """ See https://wikipedia.org/wiki/Quantile#Estimating_quantiles_from_a_sample """
    N = len(sample)
    if type == 6:  # With M = k*ert - 1 this one is exact
        h = (N+1)*p
    elif type == 7:
        h = (N-1)*p + 1
    elif type == 8:
        h = (N+1/3)*p + 1/3
    h_floor = int(h)
    if not sorted:
        sorted_sample = tf.sort(sample)
    quantile = sorted_sample[h_floor-1]
    if h_floor != h:
        quantile += (h - h_floor)*(sorted_sample[h_floor]-sorted_sample[h_floor-1])
    return quantile


def subset_matrix(
    mat: tf.Tensor, inds_0: tf.Tensor, inds_1: tf.Tensor
) -> tf.Tensor:
    subbed_rows = tf.gather(mat, inds_0, axis=0)
    subbed_rows_cols = tf.gather(subbed_rows, inds_1, axis=1)
    return subbed_rows_cols
