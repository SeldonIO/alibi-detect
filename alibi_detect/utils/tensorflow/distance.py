import numpy as np
import tensorflow as tf
from typing import Callable


def squared_pairwise_distance(x: tf.Tensor, y: tf.Tensor, a_min: float = 1e-30, a_max: float = 1e30) -> tf.Tensor:
    """
    TensorFlow pairwise squared Euclidean distance between samples x and y.

    Parameters
    ----------
    x
        Batch of instances of shape [Nx, features].
    y
        Batch of instances of shape [Ny, features].
    a_min
        Lower bound to clip distance values.
    a_max
        Upper bound to clip distance values.

    Returns
    -------
    Pairwise squared Euclidean distance [Nx, Ny].
    """
    x2 = tf.reduce_sum(x ** 2, axis=-1, keepdims=True)
    y2 = tf.reduce_sum(y ** 2, axis=-1, keepdims=True)
    dist = x2 + tf.transpose(y2, (1, 0)) - 2. * x @ tf.transpose(y, (1, 0))
    return tf.clip_by_value(dist, a_min, a_max)


def mmd2_from_kernel_matrix(kernel_mat: tf.Tensor, m: int, permute: bool = False,
                            zero_diag: bool = True) -> tf.Tensor:
    """
    Compute maximum mean discrepancy (MMD^2) between 2 samples x and y from the
    full kernel matrix between the samples.

    Parameters
    ----------
    kernel_mat
        Kernel matrix between samples x and y.
    m
        Number of instances in y.
    permute
        Whether to permute the row indices. Used for permutation tests.
    zero_diag
        Whether to zero out the diagonal of the kernel matrix.

    Returns
    -------
    MMD^2 between the samples from the kernel matrix.
    """
    n = kernel_mat.shape[0] - m
    if zero_diag:
        kernel_mat = kernel_mat - tf.linalg.diag(tf.linalg.diag_part(kernel_mat))
    if permute:
        idx = np.random.permutation(kernel_mat.shape[0])
        kernel_mat = tf.gather(tf.gather(kernel_mat, indices=idx, axis=0), indices=idx, axis=1)
    k_xx, k_yy, k_xy = kernel_mat[:-m, :-m], kernel_mat[-m:, -m:], kernel_mat[-m:, :-m]
    c_xx, c_yy = 1 / (n * (n - 1)), 1 / (m * (m - 1))
    mmd2 = c_xx * tf.reduce_sum(k_xx) + c_yy * tf.reduce_sum(k_yy) - 2. * tf.reduce_mean(k_xy)
    return mmd2


def mmd2(x: tf.Tensor, y: tf.Tensor, kernel: Callable) -> float:
    """
    Compute MMD^2 between 2 samples.

    Parameters
    ----------
    x
        Batch of instances of shape [Nx, features].
    y
        Batch of instances of shape [Ny, features].
    kernel
        Kernel function.

    Returns
    -------
    MMD^2 between the samples x and y.
    """
    n, m = x.shape[0], y.shape[0]
    c_xx, c_yy = 1 / (n * (n - 1)), 1 / (m * (m - 1))
    k_xx, k_yy, k_xy = kernel(x, x), kernel(y, y), kernel(x, y)  # type: ignore
    return (c_xx * (tf.reduce_sum(k_xx) - tf.linalg.trace(k_xx)) +
            c_yy * (tf.reduce_sum(k_yy) - tf.linalg.trace(k_yy)) - 2. * tf.reduce_mean(k_xy))


def relative_euclidean_distance(x: tf.Tensor, y: tf.Tensor, eps: float = 1e-12, axis: int = -1) -> tf.Tensor:
    """
    Relative Euclidean distance.

    Parameters
    ----------
    x
        Tensor used in distance computation.
    y
        Tensor used in distance computation.
    eps
        Epsilon added to denominator for numerical stability.
    axis
        Axis used to compute distance.

    Returns
    -------
    Tensor with relative Euclidean distance across specified axis.
    """
    denom = tf.concat([tf.reshape(tf.norm(x, ord=2, axis=axis), (-1, 1)),
                       tf.reshape(tf.norm(y, ord=2, axis=axis), (-1, 1))], axis=1)
    dist = tf.norm(x - y, ord=2, axis=axis) / (tf.reduce_min(denom, axis=axis) + eps)
    return dist
