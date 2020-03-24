from creme.utils import Histogram
import numpy as np
from typing import Dict, List
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


def build_histograms(X: np.ndarray, n_features: int, histograms: List[Histogram] = None) -> List[Histogram]:
    """
    Compute an approximate histogram to be used for calculating the ecdf.

    Parameters
    ----------
    X
        Dataset.
    n_features
        Number of features.
    histograms
        List of `creme` Histogram objects, one for each feature.
        If passed, then the histograms are updated with the new data in X and returned.

    Returns
    -------
    A list of `creme` Histogram objects, one for each feature.
    """
    X_flat = X.reshape(X.shape[0], -1)
    if X_flat.shape[-1] != n_features:
        raise ValueError('n_features does not correspond to the dimensionality of the data')

    if histograms is None:
        histograms = []
        for feature in range(n_features):
            histograms.append(Histogram())

    for x in X_flat:
        for feature in range(n_features):
            histograms[feature].update(x[feature])

    return histograms
