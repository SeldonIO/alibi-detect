import numpy as np
import random
from typing import List, Tuple
from alibi_detect.utils.data import Bunch
from alibi_detect.utils.discretizer import Discretizer
from alibi_detect.utils.distance import abdm, multidim_scaling
from alibi_detect.utils.mapping import ohe2ord


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


def inject_outlier_tabular(X: np.ndarray,
                           cols: List[int],
                           perc_outlier: int,
                           y: np.ndarray = None,
                           n_std: float = 2.,
                           min_std: float = 1.
                           ) -> Bunch:
    """
    Inject outliers in numerical tabular data.

    Parameters
    ----------
    X
        Tabular data to perturb (inject outliers).
    cols
        Columns of X that are numerical and can be perturbed.
    perc_outlier
        Percentage of observations which are perturbed to outliers. For multiple numerical features,
        the percentage is evenly split across the features.
    y
        Outlier labels.
    n_std
        Number of feature-wise standard deviations used to perturb the original data.
    min_std
        Minimum number of standard deviations away from the current observation. This is included because
        of the stochastic nature of the perturbation which could lead to minimal perturbations without a floor.

    Returns
    -------
    Bunch object with the perturbed tabular data and the outlier labels.
    """
    n_dim = len(X.shape)
    if n_dim == 1:
        X = X.reshape(-1, 1)
    n_samples, n_features = X.shape
    X_outlier = X.astype(np.float32).copy()
    if y is None:
        is_outlier = np.zeros(n_samples)
    else:
        is_outlier = y
    n_cols = len(cols)

    # distribute outliers evenly over different columns
    n_outlier = int(n_samples * perc_outlier * .01 / n_cols)
    if n_outlier == 0:
        return Bunch(data=X_outlier, target=is_outlier, target_names=['normal', 'outlier'])

    # add perturbations
    stdev = X_outlier.std(axis=0)
    for col in cols:
        outlier_idx = np.sort(random.sample(range(n_samples), n_outlier))
        rnd = np.random.normal(size=n_outlier)
        X_outlier[outlier_idx, col] += np.sign(rnd) * np.maximum(np.abs(rnd * n_std), min_std) * stdev[col]
        is_outlier[outlier_idx] = 1
    if n_dim == 1:
        X_outlier = X_outlier.reshape(n_samples, )
    return Bunch(data=X_outlier, target=is_outlier, target_names=['normal', 'outlier'])


def inject_outlier_categorical(X: np.ndarray,
                               cols: List[int],
                               perc_outlier: int,
                               y: np.ndarray = None,
                               cat_perturb: dict = None,
                               X_fit: np.ndarray = None,
                               disc_perc: list = [25, 50, 75],
                               smooth: float = 1.
                               ) -> Bunch:
    """
    Inject outliers in categorical variables of tabular data.

    Parameters
    ----------
    X
        Tabular data with categorical variables to perturb (inject outliers).
    cols
        Columns of X that are categorical and can be perturbed.
    perc_outlier
        Percentage of observations which are perturbed to outliers. For multiple numerical features,
        the percentage is evenly split across the features.
    y
        Outlier labels.
    cat_perturb
        Dictionary mapping each category in the categorical variables to their furthest neighbour.
    X_fit
        Optional data used to infer pairwise distances from.
    disc_perc
        List with percentiles used in binning of numerical features used for the 'abdm' pairwise distance measure.
    smooth
        Smoothing exponent between 0 and 1 for the distances.
        Lower values will smooth the difference in distance metric between different features.

    Returns
    -------
    Bunch object with the perturbed tabular data, outlier labels and
    a dictionary used to map categories to their furthest neighbour.
    """
    if cat_perturb is None:
        # transform the categorical variables into numerical ones via
        # pairwise distances computed with abdm and multidim scaling
        X_fit = X.copy() if X_fit is None else X_fit

        # find number of categories for each categorical variable
        cat_vars = {k: None for k in cols}
        for k in cols:
            cat_vars[k] = len(np.unique(X_fit[:, k]))  # type: ignore

        # TODO: extend method for OHE
        ohe = False
        if ohe:
            X_ord, cat_vars_ord = ohe2ord(X, cat_vars)
        else:
            X_ord, cat_vars_ord = X, cat_vars

        # bin numerical features to compute the pairwise distance matrices
        n_ord = X_ord.shape[1]
        if len(cols) != n_ord:
            fnames = [str(_) for _ in range(n_ord)]
            disc = Discretizer(X_ord, cols, fnames, percentiles=disc_perc)
            X_bin = disc.discretize(X_ord)
            cat_vars_bin = {k: len(disc.names[k]) for k in range(n_ord) if k not in cols}
        else:
            X_bin = X_ord
            cat_vars_bin = {}

        # pairwise distances for categorical variables
        d_pair = abdm(X_bin, cat_vars_ord, cat_vars_bin)

        # multidim scaling
        feature_range = (np.ones((1, n_ord)) * -1e10, np.ones((1, n_ord)) * 1e10)
        d_abs = multidim_scaling(d_pair,
                                 n_components=2,
                                 use_metric=True,
                                 standardize_cat_vars=True,
                                 smooth=smooth,
                                 feature_range=feature_range,
                                 update_feature_range=False)[0]

        # find furthest category away for each category in the categorical variables
        cat_perturb = {k: np.zeros(len(v)) for k, v in d_abs.items()}
        for k, v in d_abs.items():
            for i in range(len(v)):
                cat_perturb[k][i] = np.argmax(np.abs(v[i] - v))
    else:
        d_abs = None

    n_dim = len(X.shape)
    if n_dim == 1:
        X = X.reshape(-1, 1)
    n_samples, n_features = X.shape
    X_outlier = X.astype(np.float32).copy()
    if y is None:
        is_outlier = np.zeros(n_samples)
    else:
        is_outlier = y
    n_cols = len(cols)

    # distribute outliers evenly over different columns
    n_outlier = int(n_samples * perc_outlier * .01 / n_cols)
    for col in cols:
        outlier_idx = np.sort(random.sample(range(n_samples), n_outlier))
        col_cat = X_outlier[outlier_idx, col].astype(int)
        col_map = np.tile(cat_perturb[col], (n_outlier, 1))
        X_outlier[outlier_idx, col] = np.diag(col_map.T[col_cat])
        is_outlier[outlier_idx] = 1
    if n_dim == 1:
        X_outlier = X_outlier.reshape(n_samples, )
    return Bunch(data=X_outlier,
                 target=is_outlier,
                 cat_perturb=cat_perturb,
                 d_abs=d_abs,
                 target_names=['normal', 'outlier'])
