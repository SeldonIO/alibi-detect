from contextlib import contextmanager
import random
import numpy as np
import os
from alibi_detect.utils.frameworks import has_pytorch, has_tensorflow

if has_tensorflow:
    import tensorflow as tf
if has_pytorch:
    import torch


def set_seed(seed):
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


def get_seed() -> int:
    """
    Gets the seed set by :func:`set_seed`.

    Example
    -------
    from alibi_detect.utils import set_seed, get_seed
    set_seed(42)
    get_seed()
    >>> 42
    """
    return np.random.get_state()[1][0]  # type: ignore[index]


@contextmanager
def fixed_seed(seed: int):
    """
    A context manager to run with a requested random seed (applied to all the RNG's set by
    :func:`alibi_detect.utils.random.set_seeds`).

    Example
    -------
    set_seed(0)
    with fixed_seed(42):
        dd = cd.LSDDDrift(X_ref)  # seeds equal 42 here
        p_val = dd.predict(X_h0)['data']['p_val']
    # seeds equal 0 here

    Warning
    -------
    To ensure random seeds are reset to their original values upon exit of the context manager, it is
    recommended to use :func:`alibi_detect.utils.random.set_seed` rather than `tf.random.set_seed` etc
    individually when using this context manager.
    """
    orig_seed = get_seed()
    set_seed(seed)
    try:
        yield
    finally:
        set_seed(orig_seed)
