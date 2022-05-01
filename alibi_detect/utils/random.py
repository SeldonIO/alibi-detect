from contextlib import contextmanager
import random
import numpy as np
from alibi_detect.utils.frameworks import has_pytorch, has_tensorflow

if has_tensorflow:
    import tensorflow as tf
if has_pytorch:
    import torch


def reseed(seed):
    """
    Sets the Python, NumPy, TensorFlow and PyTorch random seeds (if installed).
    """
    random.seed(seed)
    np.random.seed(seed)
    if has_tensorflow:
        tf.random.set_seed(seed)
    if has_pytorch:
        torch.manual_seed(seed)


@contextmanager
def fixed_seed(seed: int):
    """
    A context manager to run with a requested random seed (applied to all the RNG's set by
    :func:`alibi_detect.utils.random.reseed`).

    Example
    -------
    dd = cd.MMDDrift(X_ref)
    with fixed_seed(seed):
        p_val = dd.predict(X_h0)['data']['p_val']
    """
    reseed(seed)
    yield

