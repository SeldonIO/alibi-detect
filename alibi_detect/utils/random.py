from contextlib import contextmanager
import random
import numpy as np
import os
from alibi_detect.utils.frameworks import has_pytorch, has_tensorflow

if has_tensorflow:
    import tensorflow as tf
if has_pytorch:
    import torch


def reseed(seed):
    """
    Sets the Python, NumPy, TensorFlow and PyTorch random seeds (if installed).
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
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
    reseed(0)
    with fixed_seed(42):
        dd = cd.LSDDDrift(X_ref)  # seed = 42 here
        p_val = dd.predict(X_h0)['data']['p_val']
    # seeds = 0 here

    Warning
    -------
    To ensure random seeds are reset to their original values upon exit of the context manager, it is
    recommended to use :func:`alibi_detect.utils.random.reseed` rather than `tf.random.set_seed` etc
    individually when using this context manager.
    """
    orig_seed = np.random.get_state()[1][0]  # type: ignore[index]
    reseed(seed)
    try:
        yield
    finally:
        reseed(orig_seed)
