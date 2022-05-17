"""
This submodule contains utility functions to manage random number generator (RNG) seeds. It may change
depending on how we decide to handle randomisation in tests (and elsewhere) going forwards. See
https://github.com/SeldonIO/alibi-detect/issues/250.
"""
from contextlib import contextmanager
import random
import numpy as np
import os
from alibi_detect.utils.frameworks import has_pytorch, has_tensorflow

if has_tensorflow:
    import tensorflow as tf
if has_pytorch:
    import torch

# Init global seed
_ALIBI_SEED = None


def set_seed(seed: int):
    """
    Sets the Python, NumPy, TensorFlow and PyTorch random seeds, and the PYTHONHASHSEED env variable.

    Parameters
    ----------
    seed
        Value of the random seed to set.
    """
    global _ALIBI_SEED
    seed = max(seed, 0)  # TODO: This is a fix to allow --randomly-seed=0 in setup.cfg. To be removed in future
    _ALIBI_SEED = seed
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
    if _ALIBI_SEED is not None:
        return _ALIBI_SEED
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
