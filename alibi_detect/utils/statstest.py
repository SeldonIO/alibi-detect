import numpy as np
from typing import Callable, Tuple, Union


def permutation_test(x: np.ndarray, y: np.ndarray, metric: Callable, n_permutations: int = 100,
                     **kwargs) -> Tuple[float, float, np.ndarray]:
    """
    Apply a permutation test to samples x and y.

    Parameters
    ----------
    x
        Batch of instances of shape [Nx, features].
    y
        Batch of instances of shape [Ny, features].
    n_permutations
        Number of permutations used in the test.
    metric
        Distance metric used for the test. Defaults to Maximum Mean Discrepancy.
    kwargs
        Kwargs for the metric. For the default this includes for instance the kernel used.

    Returns
    -------
    p-value obtained from the permutation test, the metric between the reference and test set
    and the metric values from the permutation test.
    """
    n, k = x.shape[0], 0
    dist = metric(x, y, **kwargs)
    x_y = np.concatenate([x, y])
    dist_permutations = np.zeros(n_permutations)
    for _ in range(n_permutations):
        np.random.shuffle(x_y)
        x, y = x_y[:n], x_y[n:]
        dist_permutation = metric(x, y, **kwargs)
        dist_permutations[_] = dist_permutation
        k += dist <= dist_permutation
    return k / n_permutations, dist, dist_permutations


def fdr(p_val: np.ndarray, q_val: float) -> Tuple[int, Union[float, np.ndarray]]:
    """
    Checks the significance of univariate tests on each variable between 2 samples of
    multivariate data via the False Discovery Rate (FDR) correction of the p-values.

    Parameters
    ----------
    p_val
        p-values for each univariate test.
    q_val
        Acceptable q-value threshold.

    Returns
    -------
    Whether any of the p-values are significant after the FDR correction
    and the max threshold value or array of potential thresholds if no p-values
    are significant.
    """
    n = p_val.shape[0]
    i = np.arange(n) + 1
    p_sorted = np.sort(p_val)
    q_threshold = q_val * i / n
    below_threshold = p_sorted < q_threshold
    try:
        idx_threshold = np.where(below_threshold)[0].max()
    except ValueError:  # sorted p-values not below thresholds
        return int(below_threshold.any()), q_threshold
    return int(below_threshold.any()), q_threshold[idx_threshold]
