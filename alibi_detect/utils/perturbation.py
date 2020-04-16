import cv2
from io import BytesIO
import numpy as np
from PIL import Image
import random
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import map_coordinates
import skimage as sk
from skimage.filters import gaussian
import tensorflow as tf
from typing import List, Tuple
from alibi_detect.utils.data import Bunch
from alibi_detect.utils.discretizer import Discretizer
from alibi_detect.utils.distance import abdm, multidim_scaling
from alibi_detect.utils.mapping import ohe2ord


def mutate_categorical(X: np.ndarray,
                       rate: float = None,
                       seed: int = 0,
                       feature_range: tuple = (0, 255)) -> tf.Tensor:
    """
    Randomly change integer feature values to values within a set range
    with a specified permutation rate.

    Parameters
    ----------
    X
        Batch of data to be perturbed.
    rate
        Permutation rate (between 0 and 1).
    seed
        Random seed.
    feature_range
        Min and max range for perturbed features.

    Returns
    -------
    Array with perturbed data.
    """
    frange = (feature_range[0] + 1, feature_range[1] + 1)
    shape = X.shape
    n_samples = np.prod(shape)
    mask = tf.random.categorical(
        tf.math.log([[1. - rate, rate]]),
        n_samples,
        seed=seed,
        dtype=tf.int32
    )
    mask = tf.reshape(mask, shape)
    possible_mutations = tf.random.uniform(
        shape,
        minval=frange[0],
        maxval=frange[1],
        dtype=tf.int32,
        seed=seed + 1
    )
    X = tf.math.floormod(tf.cast(X, tf.int32) + mask * possible_mutations, frange[1])
    return tf.cast(X, tf.float32)


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

# Note: the perturbation functions below are adopted from
# https://github.com/hendrycks/robustness/blob/master/ImageNet-C/imagenet_c/imagenet_c/corruptions.py
# and used in Dan Hendrycks and Thomas Dietterich, "Benchmarking Neural Network Robustness to Common
# Corruptions and Perturbations" (ICLR 2019).
# TODO: add proper batch support


def scale_minmax(x: np.ndarray, xrange: tuple = None) -> Tuple[np.ndarray, bool]:
    """
    Minmax scaling to [0,1].

    Parameters
    ----------
    x
        Numpy array to be scaled.
    xrange
        Tuple with min and max data range.

    Returns
    -------
    Scaled array and boolean whether the array is actually scaled.
    """
    scale_back = False
    if isinstance(xrange, tuple):
        scale_back = True
        x = (x - xrange[0]) / (xrange[1] - xrange[0])
    return x, scale_back


# Noise
def gaussian_noise(x: np.ndarray, stdev: float, xrange: tuple = None) -> np.ndarray:
    """
    Inject Gaussian noise.

    Parameters
    ----------
    x
        Instance to be perturbed.
    stdev
        Standard deviation of noise.
    xrange
        Tuple with min and max data range.

    Returns
    -------
    Perturbed instance.
    """
    x, scale_back = scale_minmax(x, xrange)
    x_gn = x + np.random.normal(size=x.shape, scale=stdev)
    if scale_back:
        x_gn = x_gn * (xrange[1] - xrange[0]) + xrange[0]
    if isinstance(xrange, tuple):
        return np.clip(x_gn, xrange[0], xrange[1])
    else:
        return x_gn


def shot_noise(x: np.ndarray, lam: float, xrange: tuple = None) -> np.ndarray:
    """
    Inject Poisson noise.

    Parameters
    ----------
    x
        Instance to be perturbed.
    lam
        Scalar for the lambda parameter determining the expectation of the interval.
    xrange
        Tuple with min and max data range.

    Returns
    -------
    Perturbed instance.
    """
    x, scale_back = scale_minmax(x, xrange)
    x_sn = np.random.poisson(x * lam) / float(lam)
    if scale_back:
        x_sn = x_sn * (xrange[1] - xrange[0]) + xrange[0]
    if isinstance(xrange, tuple):
        return np.clip(x_sn, xrange[0], xrange[1])
    else:
        return x_sn


def speckle_noise(x: np.ndarray, stdev: float, xrange: tuple = None) -> np.ndarray:
    """
    Inject speckle noise.

    Parameters
    ----------
    x
        Instance to be perturbed.
    stdev
        Standard deviation of noise.
    xrange
        Tuple with min and max data range.

    Returns
    -------
    Perturbed instance.
    """
    x, scale_back = scale_minmax(x, xrange)
    x_sp = x * (1 + np.random.normal(size=x.shape, scale=stdev))
    if scale_back:
        x_sp = x_sp * (xrange[1] - xrange[0]) + xrange[0]
    if isinstance(xrange, tuple):
        return np.clip(x_sp, xrange[0], xrange[1])
    else:
        return x_sp


def impulse_noise(x: np.ndarray, amount: float, xrange: tuple = None) -> np.ndarray:
    """
    Inject salt & pepper noise.

    Parameters
    ----------
    x
        Instance to be perturbed.
    amount
        Proportion of pixels to replace with noise.
    xrange
        Tuple with min and max data range.

    Returns
    -------
    Perturbed instance.
    """
    if isinstance(xrange, tuple):
        xmin, xmax = xrange[0], xrange[1]
    else:
        xmin, xmax = x.min(), x.max()
    x_sc = (x - xmin) / (xmax - xmin)  # scale to [0,1]
    x_in = sk.util.random_noise(x_sc, mode='s&p', amount=amount)  # inject noise
    x_in = x_in * (xmax - xmin) + xmin  # scale back
    if isinstance(xrange, tuple):
        return np.clip(x_in, xrange[0], xrange[1])
    else:
        return x_in


# Blur
def gaussian_blur(x: np.ndarray, sigma: float, multichannel: bool = True, xrange: tuple = None) -> np.ndarray:
    """
    Apply Gaussian blur.

    Parameters
    ----------
    x
        Instance to be perturbed.
    sigma
        Standard deviation determining the strength of the blur.
    multichannel
        Whether the image contains multiple channels (RGB) or not.
    xrange
        Tuple with min and max data range.

    Returns
    -------
    Perturbed instance.
    """
    x, scale_back = scale_minmax(x, xrange)
    x_gb = gaussian(x, sigma=sigma, multichannel=multichannel)
    if scale_back:
        x_gb = x_gb * (xrange[1] - xrange[0]) + xrange[0]
    if isinstance(xrange, tuple):
        return np.clip(x_gb, xrange[0], xrange[1])
    else:
        return x_gb


def clipped_zoom(x: np.ndarray, zoom_factor: float) -> np.ndarray:
    """
    Helper function for zoom blur.

    Parameters
    ----------
    x
        Instance to be perturbed.
    zoom_factor
        Zoom strength.

    Returns
    -------
    Cropped and zoomed instance.
    """
    h = x.shape[0]
    ch = int(np.ceil(h / float(zoom_factor)))  # ceil crop height(= crop width)
    top = (h - ch) // 2
    x = zoom(x[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    trim_top = (x.shape[0] - h) // 2  # trim off any extra pixels
    return x[trim_top:trim_top + h, trim_top:trim_top + h]


def zoom_blur(x: np.ndarray, max_zoom: float, step_zoom: float, xrange: tuple = None) -> np.ndarray:
    """
    Apply zoom blur.

    Parameters
    ----------
    x
        Instance to be perturbed.
    max_zoom
        Max zoom strength.
    step_zoom
        Step size to go from 1 to `max_zoom` strength.
    xrange
        Tuple with min and max data range.

    Returns
    -------
    Perturbed instance.
    """
    x, scale_back = scale_minmax(x, xrange)
    zoom_factors = np.arange(1, max_zoom, step_zoom)
    out = np.zeros_like(x)
    for zoom_factor in zoom_factors:
        out += clipped_zoom(x, zoom_factor)
    x_z = (x + out) / (len(zoom_factors) + 1)
    if scale_back:
        x_z = x_z * (xrange[1] - xrange[0]) + xrange[0]
    if isinstance(xrange, tuple):
        return np.clip(x_z, xrange[0], xrange[1])
    else:
        return x_z


def glass_blur(x: np.ndarray, sigma: float, max_delta: int, iterations: int, xrange: tuple = None) -> np.ndarray:
    """
    Apply glass blur.

    Parameters
    ----------
    x
        Instance to be perturbed.
    sigma
        Standard deviation determining the strength of the Gaussian perturbation.
    max_delta
        Maximum pixel range for the blurring.
    iterations
        Number of blurring iterations.
    xrange
        Tuple with min and max data range.

    Returns
    -------
    Perturbed instance.
    """
    nrows, ncols = x.shape[:2]

    if not isinstance(xrange, tuple):
        xrange = (x.min(), x.max())

    if xrange[0] != 0 or xrange[1] != 255:
        x = (x - xrange[0]) / (xrange[1] - xrange[0]) * 255

    x = np.uint8(gaussian(x, sigma=sigma, multichannel=True))
    for i in range(iterations):
        for h in range(nrows - max_delta, max_delta, -1):
            for w in range(ncols - max_delta, max_delta, -1):
                dx, dy = np.random.randint(-max_delta, max_delta, size=(2,))
                h_prime, w_prime = h + dy, w + dx
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]
    x_gb = gaussian(x / 255, sigma=sigma, multichannel=True)
    x_gb = x_gb * (xrange[1] - xrange[0]) + xrange[0]
    if isinstance(xrange, tuple):
        return np.clip(x_gb, xrange[0], xrange[1])
    else:
        return x_gb


def disk(radius: float, alias_blur: float = 0.1, dtype=np.float32) -> np.ndarray:
    """
    Helper function for defocus blur.

    Parameters
    ----------
    radius
        Radius for the Gaussian kernel.
    alias_blur
        Standard deviation for the Gaussian kernel in both X and Y directions.
    dtype
        Data type.

    Returns
    -------
    Kernel used for Gaussian blurring.
    """
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


def defocus_blur(x: np.ndarray, radius: int, alias_blur: float, xrange: tuple = None) -> np.ndarray:
    """
    Apply defocus blur.

    Parameters
    ----------
    x
        Instance to be perturbed.
    radius
        Radius for the Gaussian kernel.
    alias_blur
        Standard deviation for the Gaussian kernel in both X and Y directions.
    xrange
        Tuple with min and max data range.

    Returns
    -------
    Perturbed instance.
    """
    x, scale_back = scale_minmax(x, xrange)
    kernel = disk(radius=radius, alias_blur=alias_blur)
    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    x_db = np.array(channels).transpose((1, 2, 0))
    if scale_back:
        x_db = x_db * (xrange[1] - xrange[0]) + xrange[0]
    if isinstance(xrange, tuple):
        return np.clip(x_db, xrange[0], xrange[1])
    else:
        return x_db


def plasma_fractal(mapsize: int = 256, wibbledecay: float = 3.) -> np.ndarray:
    """
    Helper function to apply fog to instance.
    Generates a heightmap using diamond-square algorithm.
    Returns a square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100.

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def fog(x: np.ndarray, fractal_mult: float, wibbledecay: float, xrange: tuple = None) -> np.ndarray:
    """
    Apply fog to instance.

    Parameters
    ----------
    x
        Instance to be perturbed.
    fractal_mult
        Strength applied to `plasma_fractal` output.
    wibbledecay
        Decay factor for size of noise that is applied.
    xrange
        Tuple with min and max data range.

    Returns
    -------
    Perturbed instance.
    """
    x, scale_back = scale_minmax(x, xrange)
    max_val = x.max()
    nrows, ncols = x.shape[:2]
    x_fo = x + fractal_mult * plasma_fractal(wibbledecay=wibbledecay)[:nrows, :ncols][..., np.newaxis]
    x_fo = x_fo * max_val / (max_val + fractal_mult)
    if scale_back:
        x_fo = x_fo * (xrange[1] - xrange[0]) + xrange[0]
    if isinstance(xrange, tuple):
        return np.clip(x_fo, xrange[0], xrange[1])
    else:
        return x_fo


# Digital
def contrast(x: np.ndarray, strength: float, xrange: tuple = None) -> np.ndarray:
    """
    Change contrast of image.

    Parameters
    ----------
    x
        Instance to be perturbed.
    strength
        Strength of contrast change. Lower is actually more contrast.
    xrange
        Tuple with min and max data range.

    Returns
    -------
    Perturbed instance.
    """
    x, scale_back = scale_minmax(x, xrange)
    means = np.mean(x, axis=(0, 1), keepdims=True)
    x_co = (x - means) * strength + means
    if scale_back:
        x_co = x_co * (xrange[1] - xrange[0]) + xrange[0]
    if isinstance(xrange, tuple):
        return np.clip(x_co, xrange[0], xrange[1])
    else:
        return x_co


def brightness(x: np.ndarray, strength: float, xrange: tuple = None) -> np.ndarray:
    """
    Change brightness of image.

    Parameters
    ----------
    x
        Instance to be perturbed.
    strength
        Strength of brightness change.
    xrange
        Tuple with min and max data range.

    Returns
    -------
    Perturbed instance.
    """
    x, scale_back = scale_minmax(x, xrange)
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + strength, xrange[0], xrange[1])
    x_br = sk.color.hsv2rgb(x)
    if scale_back:
        x_br = x_br * (xrange[1] - xrange[0]) + xrange[0]
    if isinstance(xrange, tuple):
        return np.clip(x_br, xrange[0], xrange[1])
    else:
        return x_br


def saturate(x: np.ndarray, strength: tuple, xrange: tuple = None) -> np.ndarray:
    """
    Change colour saturation of image.

    Parameters
    ----------
    x
        Instance to be perturbed.
    strength
        Strength of saturation change. Tuple consists of (multiplier, shift) of the perturbation.
    xrange
        Tuple with min and max data range.

    Returns
    -------
    Perturbed instance.
    """
    x, scale_back = scale_minmax(x, xrange)
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = x[:, :, 1] * strength[0] + strength[1]
    if isinstance(xrange, tuple):
        x[:, :, 1] = np.clip(x[:, :, 1], xrange[0], xrange[1])
    x_sa = sk.color.hsv2rgb(x)
    if scale_back:
        x_sa = x_sa * (xrange[1] - xrange[0]) + xrange[0]
    if isinstance(xrange, tuple):
        return np.clip(x_sa, xrange[0], xrange[1])
    else:
        return x_sa


def pixelate(x: np.ndarray, strength: float, xrange: tuple = None) -> np.ndarray:
    """
    Change coarseness of pixels for an image.

    Parameters
    ----------
    x
        Instance to be perturbed.
    strength
        Strength of pixelation (<1). Lower is actually more pixelated.
    xrange
        Tuple with min and max data range.

    Returns
    -------
    Perturbed instance.
    """
    rows, cols = x.shape[:2]

    if not isinstance(xrange, tuple):
        xrange = (x.min(), x.max())

    if xrange[0] != 0 or xrange[1] != 255:
        x = (x - xrange[0]) / (xrange[1] - xrange[0]) * 255

    im = Image.fromarray(x.astype('uint8'), mode='RGB')
    im = im.resize((int(rows * strength), int(cols * strength)), Image.BOX)
    im = im.resize((rows, cols), Image.BOX)
    x_pi = np.array(im, dtype=np.float32) / 255
    x_pi = x_pi * (xrange[1] - xrange[0]) + xrange[0]
    return x_pi


def jpeg_compression(x: np.ndarray, strength: float, xrange: tuple = None) -> np.ndarray:
    """
    Simulate changes due to JPEG compression for an image.

    Parameters
    ----------
    x
        Instance to be perturbed.
    strength
        Strength of compression (>1). Lower is actually more compressed.
    xrange
        Tuple with min and max data range.

    Returns
    -------
    Perturbed instance.
    """
    if not isinstance(xrange, tuple):
        xrange = (x.min(), x.max())

    if xrange[0] != 0 or xrange[1] != 255:
        x = (x - xrange[0]) / (xrange[1] - xrange[0]) * 255

    x = Image.fromarray(x.astype('uint8'), mode='RGB')
    output = BytesIO()
    x.save(output, 'JPEG', quality=strength)
    x = Image.open(output)
    x_jpeg = np.array(x, dtype=np.float32) / 255
    x_jpeg = x_jpeg * (xrange[1] - xrange[0]) + xrange[0]
    return x_jpeg


def elastic_transform(x: np.ndarray, mult_dxdy: float, sigma: float,
                      rnd_rng: float, xrange: tuple = None) -> np.ndarray:
    """
    Apply elastic transformation to instance.

    Parameters
    ----------
    x
        Instance to be perturbed.
    mult_dxdy
        Multiplier for the Gaussian noise in x and y directions.
    sigma
        Standard deviation determining the strength of the Gaussian perturbation.
    rnd_rng
        Range for random uniform noise.
    xrange
        Tuple with min and max data range.

    Returns
    -------
    Perturbed instance.
    """
    x, scale_back = scale_minmax(x, xrange)
    shape = x.shape
    shape_size = shape[:2]

    mult_dxdy *= shape[0]
    sigma *= shape[0]
    rnd_rng *= shape[0]

    # random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + np.random.uniform(-rnd_rng, rnd_rng, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(x, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    dx = (gaussian(np.random.uniform(-1, 1, size=shape_size),
                   sigma, mode='reflect', truncate=3) * mult_dxdy).astype(np.float32)
    dy = (gaussian(np.random.uniform(-1, 1, size=shape_size),
                   sigma, mode='reflect', truncate=3) * mult_dxdy).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    x_et = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    if scale_back:
        x_et = x_et * (xrange[1] - xrange[0]) + xrange[0]
    if isinstance(xrange, tuple):
        return np.clip(x_et, xrange[0], xrange[1])
    else:
        return x_et
