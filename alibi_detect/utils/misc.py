import numpy as np


def quantile(sample: np.ndarray, p: float, type: int = 7,
             sorted: bool = False, interpolate: bool = True) -> float:
    """
    Estimate a desired quantile of a univariate distribution from a vector of samples

    Parameters
    ----------
    sample
        A 1D vector of values
    p
        The desired quantile in (0,1)
    type
        The method for computing the quantile.
        See https://wikipedia.org/wiki/Quantile#Estimating_quantiles_from_a_sample
    sorted
        Whether or not the vector is already sorted into ascending order
    interpolate
        Whether to interpolate the desired quantile.

    Returns
    -------
    An estimate of the quantile

    """
    N = len(sample)
    if N == 0:
        raise ValueError("Cannot compute quantiles with zero samples.")

    if len(sample.shape) != 1:
        raise ValueError("Quantile estimation only supports vectors of univariate samples.")
    if not 1/N <= p <= (N-1)/N:
        raise ValueError(f"The {p}-quantile should not be estimated using only {N} samples.")

    sorted_sample = sample if sorted else np.sort(sample)

    if type == 6:
        h = (N+1)*p
    elif type == 7:
        h = (N-1)*p + 1
    elif type == 8:
        h = (N+1/3)*p + 1/3
    else:
        raise ValueError("type must be an int with value 6, 7 or 8.")
    h_floor = int(h)
    quantile = sorted_sample[h_floor-1]
    if h_floor != h and interpolate:
        quantile += (h - h_floor)*(sorted_sample[h_floor]-sorted_sample[h_floor-1])

    return float(quantile)
