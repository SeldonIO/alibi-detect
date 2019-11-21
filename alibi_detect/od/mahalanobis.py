import logging
import numpy as np
from scipy.linalg import eigh
from typing import Dict, Union
from alibi_detect.utils.discretizer import Discretizer
from alibi_detect.utils.distance import abdm, mvdm, multidim_scaling
from alibi_detect.utils.mapping import ohe2ord, ord2num
from alibi_detect.base import BaseDetector, FitMixin, ThresholdMixin, outlier_prediction_dict

logger = logging.getLogger(__name__)

EPSILON = 1e-8


class Mahalanobis(BaseDetector, FitMixin, ThresholdMixin):

    def __init__(self,
                 threshold: float = None,
                 n_components: int = 3,
                 std_clip: int = 3,
                 start_clip: int = 100,
                 max_n: int = None,
                 cat_vars: dict = None,
                 ohe: bool = False,
                 data_type: str = 'tabular'
                 ) -> None:
        """
        Outlier detector for tabular data using the Mahalanobis distance.

        Parameters
        ----------
        threshold
            Mahalanobis distance threshold used to classify outliers.
        n_components
            Number of principal components used.
        std_clip
            Feature-wise stdev used to clip the observations before updating the mean and cov.
        start_clip
            Number of observations before clipping is applied.
        max_n
            Algorithm behaves as if it has seen at most max_n points.
        cat_vars
            Dict with as keys the categorical columns and as values
            the number of categories per categorical variable.
        ohe
            Whether the categorical variables are one-hot encoded (OHE) or not. If not OHE, they are
            assumed to have ordinal encodings.
        data_type
            Optionally specifiy the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()

        if threshold is None:
            logger.warning('No threshold level set. Need to infer threshold using `infer_threshold`.')

        self.threshold = threshold
        self.n_components = n_components
        self.std_clip = std_clip
        self.start_clip = start_clip
        self.max_n = max_n

        # variables used in mapping from categorical to numerical values
        # keys = categorical columns; values = numerical value for each of the categories
        self.cat_vars = cat_vars
        self.ohe = ohe
        self.d_abs = {}  # type: Dict

        # initial parameter values
        self.clip = None  # type: Union[None, list]
        self.mean = 0
        self.C = 0
        self.n = 0

        # set metadata
        self.meta['detector_type'] = 'online'
        self.meta['data_type'] = data_type

    def fit(self,
            X: np.ndarray,
            y: np.ndarray = None,
            d_type: str = 'abdm',
            w: float = None,
            disc_perc: list = [25, 50, 75],
            standardize_cat_vars: bool = True,
            feature_range: tuple = (-1e10, 1e10),
            smooth: float = 1.,
            center: bool = True
            ) -> None:
        """
        If categorical variables are present, then transform those to numerical values.
        This step is not necessary in the absence of categorical variables.

        Parameters
        ----------
        X
            Batch of instances used to infer distances between categories from.
        y
            Model class predictions or ground truth labels for X.
            Used for 'mvdm' and 'abdm-mvdm' pairwise distance metrics.
            Note that this is only compatible with classification problems. For regression problems,
            use the 'abdm' distance metric.
        d_type
            Pairwise distance metric used for categorical variables. Currently, 'abdm', 'mvdm' and 'abdm-mvdm'
            are supported. 'abdm' infers context from the other variables while 'mvdm' uses the model predictions.
            'abdm-mvdm' is a weighted combination of the two metrics.
        w
            Weight on 'abdm' (between 0. and 1.) distance if d_type equals 'abdm-mvdm'.
        disc_perc
            List with percentiles used in binning of numerical features used for the 'abdm'
            and 'abdm-mvdm' pairwise distance measures.
        standardize_cat_vars
            Standardize numerical values of categorical variables if True.
        feature_range
            Tuple with min and max ranges to allow for perturbed instances. Min and max ranges can be floats or
            numpy arrays with dimension (1x nb of features) for feature-wise ranges.
        smooth
            Smoothing exponent between 0 and 1 for the distances. Lower values of l will smooth the difference in
            distance metric between different features.
        center
            Whether to center the scaled distance measures. If False, the min distance for each feature
            except for the feature with the highest raw max distance will be the lower bound of the
            feature range, but the upper bound will be below the max feature range.
        """
        if self.cat_vars is None:
            raise TypeError('No categorical variables specified in the "cat_vars" argument.')

        if d_type not in ['abdm', 'mvdm', 'abdm-mvdm']:
            raise ValueError('d_type needs to be "abdm", "mvdm" or "abdm-mvdm". '
                             '{} is not supported.'.format(d_type))

        if self.ohe:
            X_ord, cat_vars_ord = ohe2ord(X, self.cat_vars)
        else:
            X_ord, cat_vars_ord = X, self.cat_vars

        # bin numerical features to compute the pairwise distance matrices
        cat_keys = list(cat_vars_ord.keys())
        n_ord = X_ord.shape[1]
        if d_type in ['abdm', 'abdm-mvdm'] and len(cat_keys) != n_ord:
            fnames = [str(_) for _ in range(n_ord)]
            disc = Discretizer(X_ord, cat_keys, fnames, percentiles=disc_perc)
            X_bin = disc.discretize(X_ord)
            cat_vars_bin = {k: len(disc.names[k]) for k in range(n_ord) if k not in cat_keys}
        else:
            X_bin = X_ord
            cat_vars_bin = {}

        # pairwise distances for categorical variables
        if d_type == 'abdm':
            d_pair = abdm(X_bin, cat_vars_ord, cat_vars_bin)
        elif d_type == 'mvdm':
            d_pair = mvdm(X_ord, y, cat_vars_ord, alpha=1)

        if (type(feature_range[0]) == type(feature_range[1]) and  # noqa
                type(feature_range[0]) in [int, float]):
            feature_range = (np.ones((1, n_ord)) * feature_range[0],
                             np.ones((1, n_ord)) * feature_range[1])

        if d_type == 'abdm-mvdm':
            # pairwise distances
            d_abdm = abdm(X_bin, cat_vars_ord, cat_vars_bin)
            d_mvdm = mvdm(X_ord, y, cat_vars_ord, alpha=1)

            # multidim scaled distances
            d_abs_abdm = multidim_scaling(d_abdm, n_components=2, use_metric=True,
                                          feature_range=feature_range,
                                          standardize_cat_vars=standardize_cat_vars,
                                          smooth=smooth, center=center,
                                          update_feature_range=False)[0]

            d_abs_mvdm = multidim_scaling(d_mvdm, n_components=2, use_metric=True,
                                          feature_range=feature_range,
                                          standardize_cat_vars=standardize_cat_vars,
                                          smooth=smooth, center=center,
                                          update_feature_range=False)[0]

            # combine abdm and mvdm
            for k, v in d_abs_abdm.items():
                self.d_abs[k] = v * w + d_abs_mvdm[k] * (1 - w)
                if center:  # center the numerical feature values
                    self.d_abs[k] -= .5 * (self.d_abs[k].max() + self.d_abs[k].min())
        else:
            self.d_abs = multidim_scaling(d_pair, n_components=2, use_metric=True,
                                          feature_range=feature_range,
                                          standardize_cat_vars=standardize_cat_vars,
                                          smooth=smooth, center=center,
                                          update_feature_range=False)[0]

    def infer_threshold(self,
                        X: np.ndarray,
                        threshold_perc: float = 95.
                        ) -> None:
        """
        Update threshold by a value inferred from the percentage of instances considered to be
        outliers in a sample of the dataset.

        Parameters
        ----------
        X
            Batch of instances.
        threshold_perc
            Percentage of X considered to be normal based on the outlier score.
        """
        # convert categorical variables to numerical values
        X = self.cat2num(X)

        # compute outlier scores
        iscore = self.score(X)

        # update threshold
        self.threshold = np.percentile(iscore, threshold_perc)

    def cat2num(self, X: np.ndarray) -> np.ndarray:
        """
        Convert categorical variables to numerical values.

        Parameters
        ----------
        X
            Batch of instances to analyze.

        Returns
        -------
        Batch of instances where categorical variables are converted to numerical values.
        """
        if self.cat_vars is not None:  # convert categorical variables
            if self.ohe:
                X = ohe2ord(X, self.cat_vars)[0]
            X = ord2num(X, self.d_abs)
        return X

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute outlier scores.

        Parameters
        ----------
        X
            Batch of instances to analyze.

        Returns
        -------
        Array with outlier scores for each instance in the batch.
        """
        n_batch, n_params = X.shape  # batch size and number of features
        n_components = min(self.n_components, n_params)
        if self.max_n is not None:
            n = min(self.n, self.max_n)  # n can never be above max_n
        else:
            n = self.n

        # clip X
        if self.n > self.start_clip:
            X_clip = np.clip(X, self.clip[0], self.clip[1])
        else:
            X_clip = X

        # track mean and covariance matrix
        roll_partial_means = X_clip.cumsum(axis=0) / (np.arange(n_batch) + 1).reshape((n_batch, 1))
        coefs = (np.arange(n_batch) + 1.) / (np.arange(n_batch) + n + 1.)
        new_means = self.mean + coefs.reshape((n_batch, 1)) * (roll_partial_means - self.mean)
        new_means_offset = np.empty_like(new_means)
        new_means_offset[0] = self.mean
        new_means_offset[1:] = new_means[:-1]

        coefs = ((n + np.arange(n_batch)) / (n + np.arange(n_batch) + 1.)).reshape((n_batch, 1, 1))
        B = coefs * np.matmul((X_clip - new_means_offset)[:, :, None], (X_clip - new_means_offset)[:, None, :])
        cov_batch = (n - 1.) / (n + max(1, n_batch - 1.)) * self.C + 1. / (n + max(1, n_batch - 1.)) * B.sum(axis=0)

        # PCA
        eigvals, eigvects = eigh(cov_batch, eigvals=(n_params - n_components, n_params - 1))

        # projections
        proj_x = np.matmul(X, eigvects)
        proj_x_clip = np.matmul(X_clip, eigvects)
        proj_means = np.matmul(new_means_offset, eigvects)
        if type(self.C) == int and self.C == 0:
            proj_cov = np.diag(np.zeros(n_components))
        else:
            proj_cov = np.matmul(eigvects.transpose(), np.matmul(self.C, eigvects))

        # outlier scores are computed in the principal component space
        coefs = (1. / (n + np.arange(n_batch) + 1.)).reshape((n_batch, 1, 1))
        B = coefs * np.matmul((proj_x_clip - proj_means)[:, :, None], (proj_x_clip - proj_means)[:, None, :])
        all_C_inv = np.zeros_like(B)
        c_inv = None
        for i, b in enumerate(B):
            if c_inv is None:
                if abs(np.linalg.det(proj_cov)) > EPSILON:
                    c_inv = np.linalg.inv(proj_cov)
                    all_C_inv[i] = c_inv
                    continue
                else:
                    if n + i == 0:
                        continue
                    proj_cov = (n + i - 1.) / (n + i) * proj_cov + b
                    continue
            else:
                c_inv = (n + i - 1.) / float(n + i - 2.) * all_C_inv[i - 1]
            BC1 = np.matmul(B[i - 1], c_inv)
            all_C_inv[i] = c_inv - 1. / (1. + np.trace(BC1)) * np.matmul(c_inv, BC1)

        # update parameters
        self.mean = new_means[-1]
        self.C = cov_batch
        stdev = np.sqrt(np.diag(cov_batch))
        self.n += n_batch
        if self.n > self.start_clip:
            self.clip = [self.mean - self.std_clip * stdev, self.mean + self.std_clip * stdev]

        # compute outlier scores
        x_diff = proj_x - proj_means
        outlier_score = np.matmul(x_diff[:, None, :], np.matmul(all_C_inv, x_diff[:, :, None])).reshape(n_batch)
        return outlier_score

    def predict(self,
                X: np.ndarray,
                return_instance_score: bool = True) \
            -> Dict[Dict[str, str], Dict[np.ndarray, np.ndarray]]:
        """
        Compute outlier scores and transform into outlier predictions.

        Parameters
        ----------
        X
            Batch of instances.
        return_instance_score
            Whether to return instance level outlier scores.

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the outlier predictions and instance level outlier scores.
        """
        # convert categorical variables to numerical values
        X = self.cat2num(X)

        # compute outlier scores
        iscore = self.score(X)

        # values above threshold are outliers
        outlier_pred = (iscore > self.threshold).astype(int)

        # populate output dict
        od = outlier_prediction_dict()
        od['meta'] = self.meta
        od['data']['is_outlier'] = outlier_pred
        if return_instance_score:
            od['data']['instance_score'] = iscore
        return od
