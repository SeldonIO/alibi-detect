import logging
import numpy as np
from typing import Callable, Dict, Tuple, Optional, Union
from alibi_detect.cd.base import BaseUnivariateDrift
from scipy.stats import hypergeom
import numba as nb
import math

logger = logging.getLogger(__name__)


class FETDrift(BaseUnivariateDrift):
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            p_val: float = .05,
            preprocess_x_ref: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            correction: str = 'bonferroni',
            alternative: str = 'decrease',
            n_features: Optional[int] = None,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Fisher exact test (FET) data drift detector, which tests for a change in the mean of binary data.
        For multivariate data, the Bonferroni or False Discovery Rate (FDR) correction is applied.

        Parameters
        ----------
        x_ref
            Data used as reference distribution. Data must consist of either [True, False]'s, or [0, 1]'s.
        p_val
            p-value used for significance of the FET test. If the FDR correction method
            is used, this corresponds to the acceptable q-value.
        preprocess_x_ref
            Whether to already preprocess and store the reference data.
        update_x_ref
            Reference data can optionally be updated to the last n instances seen by the detector
            or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while
            for reservoir sampling {'reservoir_sampling': n} is passed.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        correction
            Correction type for multivariate data. Either 'bonferroni' or 'fdr' (False Discovery Rate).
        alternative
            Defines the alternative hypothesis. Options are 'decrease', 'increase' or 'change', corresponding to
            a decrease, increase, or any change in the mean.
        n_features
            Number of features used in the FET test. No need to pass it if no preprocessing takes place.
            In case of a preprocessing step, this can also be inferred automatically but could be more
            expensive to compute.
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__(
            x_ref=x_ref,
            p_val=p_val,
            preprocess_x_ref=preprocess_x_ref,
            update_x_ref=update_x_ref,
            preprocess_fn=preprocess_fn,
            correction=correction,
            n_features=n_features,
            input_shape=input_shape,
            data_type=data_type
        )
        if alternative.lower() not in ['decrease', 'increase', 'change']:
            raise ValueError("`alternative` must be either 'decrease', 'increase' or 'change'.")
        self.alternative = alternative.lower()

        # Check data is only [False, True] or [0, 1]
        values = set(np.unique(x_ref))
        if not set(values).issubset(['0', '1', True, False]):
            raise ValueError("The `x_ref` data must consist of only (0,1)'s or (False,True)'s for the "
                             "FETDrift detector.")

    def feature_score(self, x_ref: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs Fisher exact test(s), computing the p-value per feature.

        Parameters
        ----------
        x_ref
            Reference instances to compare distribution with. Data must consist of either [True, False]'s, or [0, 1]'s.
        x
            Batch of instances. Data must consist of either [True, False]'s, or [0, 1]'s.

        Returns
        -------
        Feature level p-values and odds ratios.
        """
        x = x.reshape(x.shape[0], -1).astype(dtype=np.int64)
        x_ref = x_ref.reshape(x_ref.shape[0], -1).astype(dtype=np.int64)

        # Check data is only [False, True] or [0, 1]
        values = set(np.unique(x))
        if not set(values).issubset(['0', '1', True, False]):
            raise ValueError("The `x` data must consist of only [0,1]'s or [False,True]'s for the FETDrift detector.")

        # Apply test per feature
        n = x.shape[0]
        n_ref = x_ref.shape[0]
        sum_ref = np.sum(x_ref, axis=0)
        sum_test = np.sum(x, axis=0)
        a = sum_test
        b = np.full_like(a, sum_ref)
        c = n - sum_test
        d = np.full_like(c, n_ref - sum_ref)
        odds_ratio, p_val = fisher_exact(a, b, c, d, alternative=self.alternative)

        return p_val, odds_ratio


def fisher_exact(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, alternative: str = 'decrease') \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    A vectorised implementation of scipy.stats.fisher_exact.
    """
    if any(np.any(cell) < 0 for cell in [a, b, c, d]):
        raise ValueError("All values in contingency table must be non-negative.")

    # Init arrays
    pvalue = np.full_like(a, 1.1, dtype=np.float64)
    oddsratio = np.zeros_like(a, dtype=np.float64)

    # If any rows or cols add up to zero, return p value of 1 (done later with zero_idx index)
    # If the above, or if c>0 and b>0 NOT satisfied, then set oddsratio to np.nan (via nan_idx)
    idx_zero = (a + b == 0) | (c + d == 0) | (a + c == 0) | (b + d == 0)
    idx_nan = ~((c > 0) & (b > 0))
    idx_nan = np.logical_or(idx_zero, idx_nan)
    oddsratio[idx_nan] = np.nan
    oddsratio[~idx_nan] = a[~idx_nan] * d[~idx_nan] / np.float64(b[~idx_nan] * c[~idx_nan])

    # Sums
    n1 = a + b
    n2 = c + d
    n = a + c

    # Compute p value
    if alternative == 'decrease':
        pvalue = hypergeom.cdf(a, n1 + n2, n1, n)

    elif alternative == 'increase':
        # Same formula as the 'decrease' case, but with the second column.
        pvalue = hypergeom.cdf(b, n1 + n2, n1, b + d)

    elif alternative == 'change':
        mode = np.int64(np.float64((n + 1) * (n1 + 1)) / (n1 + n2 + 2))
        pexact = hypergeom.pmf(a, n1 + n2, n1, n)
        pmode = hypergeom.pmf(mode, n1 + n2, n1, n)

        eps = 1 - 1e-4  # stick with scipy epsilon here
        idx_lt_1meps = np.where(np.abs(pexact - pmode) / np.maximum(pexact, pmode) <= 1 - eps)
        pvalue[idx_lt_1meps] = 1.

        idx_lt_mode = (a < mode)
        idx_gt_pexact = np.logical_and(a < mode, hypergeom.pmf(n, n1 + n2, n1, n) > pexact / eps)
        idx_lt_mode = np.logical_and(idx_lt_mode, ~idx_gt_pexact)
        plower = hypergeom.cdf(a, n1 + n2, n1, n)
        guess = _binary_search(n[idx_lt_mode], n1[idx_lt_mode], n2[idx_lt_mode],
                               mode[idx_lt_mode], pexact[idx_lt_mode], True, eps)
        pvalue[idx_lt_mode] = plower[idx_lt_mode] + (1-hypergeom.cdf(guess - 1, n1[idx_lt_mode] + n2[idx_lt_mode],
                                                                     n1[idx_lt_mode], n[idx_lt_mode]))  # TODO sf=1-cdf
        pvalue[idx_gt_pexact] = plower[idx_gt_pexact]

        idx_remain = (pvalue > 1.)
        idx_gt_pexact = np.logical_and(pvalue > 1., hypergeom.pmf(0, n1 + n2, n1, n) > pexact / eps)
        idx_remain = np.logical_and(idx_remain, ~idx_gt_pexact)
        pupper = 1 - hypergeom.cdf(a - 1, n1 + n2, n1, n)  # TODO, replace with cdf once sf issue resolved
        guess = _binary_search(n[idx_remain], n1[idx_remain], n2[idx_remain],
                               mode[idx_remain], pexact[idx_remain], False, eps)
        pvalue[idx_remain] = pupper[idx_remain] + hypergeom.cdf(guess, n1[idx_remain] + n2[idx_remain],
                                                                n1[idx_remain], n[idx_remain])
        pvalue[idx_gt_pexact] = pupper[idx_gt_pexact]

    # Limit pvalue to 1
    pvalue = np.minimum(pvalue, 1.0)
    pvalue[idx_zero] = 1.0

    return oddsratio, pvalue


@nb.njit(cache=True)
def _betaln(a: Union[int, float], b: Union[int, float]) -> float:
    """
    Natural logarithm of absolute value of beta function. Equivalent to scipy.special.betaln, but numba accelerated.

    Parameters
    ----------
    a
        Parameter a in beta(a,b)
    b
        Parameter b in beta(a,b)

    Returns
    -------
    The result of ln(abs(beta(a,b))).
    """
    if a < 0 or b < 0:
        raise ValueError("betaln only implemented for positive values.")
    elif a == 0 or b == 0:
        return np.inf
    beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a+b)
    return beta


@nb.njit(cache=True)
def _pmf(k: int, M: int, n: int, N: int) -> float:
    """
    Compute the probability mass function of a hypergeom distribution. Similar to scipy's hypergeom.pmf, but
    numba accelerated in order to work in the numba vectorised _binary_search.
    """
    tot, good = M, n
    bad = tot - good
    logpmf = _betaln(good + 1, 1) + _betaln(bad + 1, 1) + _betaln(tot - N + 1, N + 1) - \
        _betaln(k + 1, good - k + 1) - _betaln(N - k + 1, bad - N + k + 1) - _betaln(tot + 1, 1)
    return np.exp(logpmf)


@nb.vectorize(
    [nb.int64(nb.int64, nb.int64, nb.int64, nb.int64, nb.float64, nb.boolean, nb.float64)],
    target='parallel', cache=True)
def _binary_search(n: int, n1: int, n2: int, mode: int, pexact: float, upper: bool, eps: float) -> int:
    """Binary search for where to begin lower/upper halves in two-sided test.

    Based on binary_search in:
    https://github.com/scipy/scipy/blob/47bb6febaa10658c72962b9615d5d5aa2513fa3a/scipy/stats/stats.py
    """
    if upper:
        minval = mode
        maxval = n
    else:
        minval = 0
        maxval = mode
    guess = -1
    while maxval - minval > 1:
        if maxval == minval + 1 and guess == minval:
            guess = maxval
        else:
            guess = (maxval + minval) // 2
        pguess = _pmf(guess, n1 + n2, n1, n)
        if upper:
            ng = guess - 1
        else:
            ng = guess + 1
        if pguess <= pexact < _pmf(ng, n1 + n2, n1, n):
            break
        elif pguess < pexact:
            maxval = guess
        else:
            minval = guess
    if guess == -1:
        guess = minval
    if upper:
        while guess > 0 and _pmf(guess, n1 + n2, n1, n) < pexact * eps:
            guess -= 1
        while _pmf(guess, n1 + n2, n1, n) > pexact / eps:
            guess += 1
    else:
        while _pmf(guess, n1 + n2, n1, n) < pexact * eps:
            guess += 1
        while guess > 0 and _pmf(guess, n1 + n2, n1, n) > pexact / eps:
            guess -= 1
    return guess
