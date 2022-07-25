import numpy as np
import tensorflow as tf


def mutate_categorical(X: np.ndarray,
                       rate: float = None,
                       seed: int = 0,
                       feature_range: tuple = (0, 255)) -> tf.Tensor:
    """
    Randomly change integer feature values to values within a set range
    with a specified permutation rate.

    Parameters
    ----------
    X
        Batch of data to be perturbed.
    rate
        Permutation rate (between 0 and 1).
    seed
        Random seed.
    feature_range
        Min and max range for perturbed features.

    Returns
    -------
    Array with perturbed data.
    """
    frange = (feature_range[0] + 1, feature_range[1] + 1)
    shape = X.shape
    n_samples = np.prod(shape)
    mask = tf.random.categorical(
        tf.math.log([[1. - rate, rate]]),
        n_samples,
        seed=seed,
        dtype=tf.int32
    )
    mask = tf.reshape(mask, shape)
    possible_mutations = tf.random.uniform(
        shape,
        minval=frange[0],
        maxval=frange[1],
        dtype=tf.int32,
        seed=seed + 1
    )
    X = tf.math.floormod(tf.cast(X, tf.int32) + mask * possible_mutations, frange[1])
    return tf.cast(X, tf.float32)
