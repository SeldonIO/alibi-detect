import numpy as np
import random


def reservoir_sampling(X_ref: np.ndarray,
                       X: np.ndarray,
                       reservoir_size: int,
                       n: int) -> np.ndarray:
    """
    Apply reservoir sampling.

    Parameters
    ----------
    X_ref
        Current instances in reservoir.
    X
        Data to update reservoir with.
    reservoir_size
        Size of reservoir.
    n
        Number of total instances that have passed so far.

    Returns
    -------
    Updated reservoir.
    """
    if X.shape[0] + n <= reservoir_size:
        return np.concatenate([X_ref, X], axis=0)

    n_ref = X_ref.shape[0]
    output_size = min(reservoir_size, n_ref + X.shape[0])
    shape = (output_size,) + X.shape[1:]
    X_reservoir = np.zeros(shape, dtype=X_ref.dtype)
    X_reservoir[:n_ref] = X_ref
    for item in X:
        n += 1
        if n_ref < reservoir_size:
            X_reservoir[n_ref, :] = item
            n_ref += 1
        else:
            r = int(random.random() * n)
            if r < reservoir_size:
                X_reservoir[r, :] = item
    return X_reservoir
