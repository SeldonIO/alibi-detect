import numpy as np
import random
from typing import Tuple
from alibi_detect.utils.data import Bunch


def apply_mask(X: np.ndarray,
               mask_size: tuple = (4, 4),
               n_masks: int = 1,
               coord: tuple = None,
               channels: list = [0, 1, 2],
               mask_type: str = 'uniform',
               noise_distr: tuple = (0, 1),
               noise_rng: tuple = (0, 1),
               clip_rng: tuple = (0, 1)
               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mask images. Can zero out image patches or add normal or uniformly distributed noise.

    Parameters
    ----------
    X
        Batch of instances to be masked.
    mask_size
        Tuple with the size of the mask.
    n_masks
        Number of masks applied for each instance in the batch X.
    coord
        Upper left (x,y)-coordinates for the mask.
    channels
        Channels of the image to apply the mask to.
    mask_type
        Type of mask. One of 'uniform', 'random' (both additive noise) or 'zero' (zero values for mask).
    noise_distr
        Mean and standard deviation for noise of 'random' mask type.
    noise_rng
        Min and max value for noise of 'uniform' type.
    clip_rng
        Min and max values for the masked instances.

    Returns
    -------
    Tuple with masked instances and the masks.
    """
    X_shape = X.shape

    # initialize mask
    if mask_type != 'zero':
        mask = np.zeros((n_masks,) + X_shape[1:])
    elif mask_type == 'zero':
        mask = np.ones((n_masks,) + X_shape[1:])
    else:
        raise ValueError('Only `normal`, `uniform` and `zero` masking available.')

    # create noise for mask
    if mask_type == 'normal':
        noise = np.random.normal(loc=noise_distr[0], scale=noise_distr[1], size=(n_masks,) + mask_size)
    elif mask_type == 'uniform':
        noise = np.random.uniform(low=noise_rng[0], high=noise_rng[1], size=(n_masks,) + mask_size)

    # find upper left coordinate for mask
    if coord is None:
        x_start = np.random.randint(0, X_shape[1] - mask_size[0], n_masks)
        y_start = np.random.randint(0, X_shape[2] - mask_size[1], n_masks)
    else:
        x_start, y_start = coord

    # update masks
    for _ in range(x_start.shape[0]):

        if mask_type == 'zero':
            update_val = 0
        else:
            update_val = noise[_]

        for c in channels:
            mask[
                _,
                x_start[_]:x_start[_] + mask_size[0],
                y_start[_]:y_start[_] + mask_size[1],
                c
            ] = update_val

    # apply masks to instances
    X_mask = []
    for _ in range(X_shape[0]):
        if mask_type == 'zero':
            X_mask_ = X[_].reshape((1,) + X_shape[1:]) * mask
        else:
            X_mask_ = np.clip(X[_].reshape((1,) + X_shape[1:]) + mask, clip_rng[0], clip_rng[1])
        X_mask.append(X_mask_)
    X_mask = np.concatenate(X_mask, axis=0)

    return X_mask, mask


def inject_outlier_ts(X: np.ndarray,
                      perc_outlier: int,
                      perc_window: int = 10,
                      n_std: float = 2.,
                      min_std: float = 1.
                      ) -> Bunch:
    """
    Inject outliers in both univariate and multivariate time series data.

    Parameters
    ----------
    X
        Time series data to perturb (inject outliers).
    perc_outlier
        Percentage of observations which are perturbed to outliers. For multivariate data,
        the percentage is evenly split across the individual time series.
    perc_window
        Percentage of the observations used to compute the standard deviation used in the perturbation.
    n_std
        Number of standard deviations in the window used to perturb the original data.
    min_std
        Minimum number of standard deviations away from the current observation. This is included because
        of the stochastic nature of the perturbation which could lead to minimal perturbations without a floor.

    Returns
    -------
    Bunch object with the perturbed time series and the outlier labels.
    """
    n_dim = len(X.shape)
    if n_dim == 1:
        X = X.reshape(-1, 1)
    n_samples, n_ts = X.shape
    X_outlier = X.copy()
    is_outlier = np.zeros(n_samples)
    # one sided window used to compute mean and stdev from
    window = int(perc_window * n_samples * .5 / 100)
    # distribute outliers evenly over different time series
    n_outlier = int(n_samples * perc_outlier * .01 / n_ts)
    if n_outlier == 0:
        return Bunch(data=X_outlier, target=is_outlier, target_names=['normal', 'outlier'])
    for s in range(n_ts):
        outlier_idx = np.sort(random.sample(range(n_samples), n_outlier))
        window_idx = [
            np.maximum(outlier_idx - window, 0),
            np.minimum(outlier_idx + window, n_samples)
        ]
        stdev = np.array([X_outlier[window_idx[0][i]:window_idx[1][i], s].std() for i in range(len(outlier_idx))])
        rnd = np.random.normal(size=n_outlier)
        X_outlier[outlier_idx, s] += np.sign(rnd) * np.maximum(np.abs(rnd * n_std), min_std) * stdev
        is_outlier[outlier_idx] = 1
    if n_dim == 1:
        X_outlier = X_outlier.reshape(n_samples,)
    return Bunch(data=X_outlier, target=is_outlier, target_names=['normal', 'outlier'])
