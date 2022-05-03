from contextlib import contextmanager
import random
import numpy as np
import os
import sys
from alibi_detect.utils.frameworks import has_pytorch, has_tensorflow

# Pointer to the alibi_detect module. Used to set a global seed variable.
_module_pointer = sys.modules[__name__]
setattr(_module_pointer, '_seed', None)

if has_tensorflow:
    import tensorflow as tf
if has_pytorch:
    import torch


def set_seed(seed: int):
    """
    Sets the Python, NumPy, TensorFlow and PyTorch random seeds (if installed).

    Parameters
    ----------
    seed
        Value of the random seed to set.
    """
    setattr(_module_pointer, '_seed', seed)
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
    >>> from alibi_detect.utils._random import set_seed, get_seed
    >>> set_seed(42)
    >>> get_seed()
    42
    """
    if _module_pointer._seed is not None:
        return _module_pointer._seed
    else:
        raise RuntimeError('`set_seed` must be called before `get_seed` can be called.')


@contextmanager
def fixed_seed(seed: int):
    """
    A context manager to run with a requested random seed (applied to all the RNG's set by :func:`set_seed`).

    Parameters
    ----------
    seed
        Value of the random seed to set in the isolated context.

    Example
    -------
    .. code-block :: python

        set_seed(0)
        with fixed_seed(42):
            dd = cd.LSDDDrift(X_ref)  # seeds equal 42 here
            p_val = dd.predict(X_h0)['data']['p_val']
        # seeds equal 0 here
    """
    orig_seed = get_seed()
    set_seed(seed)
    try:
        yield
    finally:
        set_seed(orig_seed)
