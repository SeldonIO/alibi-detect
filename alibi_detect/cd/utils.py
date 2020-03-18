import numpy as np
from typing import Dict
from alibi_detect.utils.sampling import reservoir_sampling


def update_reference(X_ref: np.ndarray,
                     X: np.ndarray,
                     n: int,
                     update_method: Dict[str, int] = None,
                     ) -> np.ndarray:
    """
    Update reference dataset for drift detectors.

    Parameters
    ----------
    X_ref
        Current reference dataset.
    X
        New data.
    n
        Count of the total number of instances that have been used so far.
    update_method
        Dict with as key `reservoir_sampling` or `last` and as value n. `reservoir_sampling` will apply
        reservoir sampling with reservoir of size n while `last` will return (at most) the last n instances.

    Returns
    -------
    Updated reference dataset.
    """
    if isinstance(update_method, dict):
        update_type = list(update_method.keys())[0]
        size = update_method[update_type]
        if update_type == 'reservoir_sampling':
            return reservoir_sampling(X_ref, X, size, n)
        elif update_type == 'last':
            X_update = np.concatenate([X_ref, X], axis=0)
            return X_update[-size:]
        else:
            raise KeyError('Only `reservoir_sampling` and `last` are valid update options for X_ref.')
    else:
        return X_ref


def fdr(p_val: np.ndarray, q_val: float) -> bool:
    """
    Checks the significance of univariate tests on each variable between 2 samples of multivariate data
    via the False Discovery Rate (FDR) correction of the p-values.

    Parameters
    ----------
    p_val
        p-values for each univariate test.
    q_val
        Acceptable q-value threshold.

    Returns
    -------
    Whether any of the p-values are significant after the FDR correction.
    """
    n = p_val.shape[0]
    i = np.arange(n) + 1
    p_sorted = np.sort(p_val)
    q_threshold = q_val * i / n
    below_threshold = (p_sorted < q_threshold).any()
    return below_threshold
