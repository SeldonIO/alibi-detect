import dask.array as da
import numpy as np
from typing import Callable, Tuple, Union
from alibi_detect.utils.distance import maximum_mean_discrepancy


def permutation_test(x: Union[np.ndarray, da.array],
                     y: Union[np.ndarray, da.array],
                     n_permutations: int = 1000,
                     metric: Callable = maximum_mean_discrepancy,
                     return_distance: bool = False,
                     return_permutation_distance: bool = False,
                     **kwargs) \
        -> Union[np.float, Tuple[np.float, np.float], Tuple[np.float, np.float, np.ndarray]]:
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
    return_distance
        Whether to return the test statistic.
    return_permutation_distance
        Whether to return the test statistics for each permutation.
    kwargs
        Kwargs for the metric. For the default this includes for instance the kernel used.

    Returns
    -------
    p-value obtained from the test.
    """
    is_np = True if isinstance(x, np.ndarray) and isinstance(y, np.ndarray) else False
    n, k = x.shape[0], 0
    dist = metric(x, y, **kwargs)
    x_y = np.concatenate([x, y])
    if not is_np:  # dask array
        dist = dist.compute()
        xchunks, ychunks = x.chunksize, y.chunksize
        x_y = np.array(x_y)
    dist_permutations = np.zeros(n_permutations)
    for _ in range(n_permutations):
        np.random.shuffle(x_y)
        x, y = x_y[:n], x_y[n:]
        if not is_np:
            x, y = da.from_array(x, chunks=xchunks), da.from_array(y, chunks=ychunks)
        dist_permutation = metric(x, y, **kwargs)
        if not is_np:
            dist_permutation = dist_permutation.compute()
        dist_permutations[_] = dist_permutation
        k += dist <= dist_permutation
    outputs = (k / n_permutations,)
    if return_distance:
        outputs += (dist,)  # type: ignore
    if return_permutation_distance:
        outputs += (dist_permutations,)  # type: ignore
    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs


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
