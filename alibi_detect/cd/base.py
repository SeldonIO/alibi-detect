from abc import abstractmethod
import logging
import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy.stats import binom_test, ks_2samp
from typing import Callable, Dict, List, Optional, Tuple, Union
from alibi_detect.base import BaseDetector, concept_drift_dict
from alibi_detect.cd.utils import update_reference
from alibi_detect.utils.frameworks import has_pytorch, has_tensorflow
from alibi_detect.utils.statstest import fdr

if has_pytorch:
    import torch  # noqa F401

if has_tensorflow:
    import tensorflow as tf  # noqa F401

logger = logging.getLogger(__name__)


class BaseClassifierDrift(BaseDetector):
    def __init__(
            self,
            x_ref: np.ndarray,
            p_val: float = .05,
            preprocess_x_ref: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            preds_type: str = 'probs',
            binarize_preds: bool = False,
            train_size: Optional[float] = .75,
            n_folds: Optional[int] = None,
            seed: int = 0,
            data_type: Optional[str] = None
    ) -> None:
        """
        Base class for the classifier-based drift detector.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        p_val
            p-value used for the significance of the test.
        preprocess_x_ref
            Whether to already preprocess and store the reference data.
        update_x_ref
            Reference data can optionally be updated to the last n instances seen by the detector
            or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while
            for reservoir sampling {'reservoir_sampling': n} is passed.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        preds_type
            Whether the model outputs probabilities or logits
        binarize_preds
            Whether to test for discrepency on soft (e.g. probs/logits) model predictions directly
            with a K-S test or binarise to 0-1 prediction errors and apply a binomial test.
        train_size
            Optional fraction (float between 0 and 1) of the dataset used to train the classifier.
            The drift is detected on `1 - train_size`. Cannot be used in combination with `n_folds`.
        n_folds
            Optional number of stratified folds used for training. The model preds are then calculated
            on all the out-of-fold predictions. This allows to leverage all the reference and test data
            for drift detection at the expense of longer computation. If both `train_size` and `n_folds`
            are specified, `n_folds` is prioritized.
        seed
            Optional random seed for fold selection.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()

        if p_val is None:
            logger.warning('No p-value set for the drift threshold. Need to set it to detect data drift.')

        if isinstance(train_size, float) and isinstance(n_folds, int):
            logger.warning('Both `n_folds` and `train_size` specified. By default `n_folds` is used.')

        if preds_type not in ['probs', 'logits']:
            raise ValueError("'preds_type' should be 'probs' or 'logits'")

        # optionally already preprocess reference data
        self.p_val = p_val
        if preprocess_x_ref and isinstance(preprocess_fn, Callable):  # type: ignore
            self.x_ref = preprocess_fn(x_ref)
        else:
            self.x_ref = x_ref
        self.preprocess_x_ref = preprocess_x_ref
        self.update_x_ref = update_x_ref
        self.preprocess_fn = preprocess_fn
        self.n = x_ref.shape[0]  # type: ignore

        # define whether soft preds and optionally the stratified k-fold split
        self.preds_type = preds_type
        self.binarize_preds = binarize_preds
        if isinstance(n_folds, int):
            self.train_size = None
            self.skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        else:
            self.train_size, self.skf = train_size, None

        # set metadata
        self.meta['detector_type'] = 'offline'
        self.meta['data_type'] = data_type
        self.meta['params'] = {'binarize_preds ': binarize_preds, 'preds_type': preds_type}

    def preprocess(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Data preprocessing before computing the drift scores.
        Parameters
        ----------
        x
            Batch of instances.
        Returns
        -------
        Preprocessed reference data and new instances.
        """
        if isinstance(self.preprocess_fn, Callable):  # type: ignore
            x = self.preprocess_fn(x)
            x_ref = self.x_ref if self.preprocess_x_ref else self.preprocess_fn(self.x_ref)
            return x_ref, x
        else:
            return self.x_ref, x

    def get_splits(self, x_ref: np.ndarray, x: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Split reference and test data in train and test folds used by the classifier.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        x
            Batch of instances.

        Returns
        -------
        List with tuples of train and test indices for optionally different folds.
        """
        # create dataset and labels
        y = np.concatenate([np.zeros(x_ref.shape[0]), np.ones(x.shape[0])], axis=0).astype(int)
        x = np.concatenate([x_ref, x], axis=0)

        # random shuffle if stratified folds are not used
        if self.skf is None:
            n_tot = x.shape[0]
            idx_shuffle = np.random.choice(np.arange(x.shape[0]), size=n_tot, replace=False)
            n_tr = int(n_tot * self.train_size)
            idx_tr, idx_te = idx_shuffle[:n_tr], idx_shuffle[n_tr:]
            splits = [(idx_tr, idx_te)]
        else:  # use stratified folds
            splits = self.skf.split(x, y)
        return x, y, splits

    def test_probs(
        self, y_oof: np.ndarray, probs_oof: np.ndarray, n_ref: int, n_cur: int
    ) -> Tuple[float, float]:
        """
        Perform a statistical test of the probabilities predicted by the model against
        what we'd expect under the no-change null.

        Parameters
        ----------
        y_oof
            Out of fold targets (0 ref, 1 cur)
        probs_oof
            Probabilities predicted by the model
        n_ref
            Size of reference window used in training model
        n_cur
            Size of current window used in trianing model

        Returns
        -------
        p-value and notion of performance of classifier relative to expectation under null
        """
        probs_oof = probs_oof[:, 1]  # [1-p, p]

        if self.binarize_preds:
            baseline_accuracy = max(n_ref, n_cur) / (n_ref + n_cur)  # exp under null
            n_oof = y_oof.shape[0]
            n_correct = (y_oof == probs_oof.round()).sum()
            p_val = binom_test(n_correct, n_oof, baseline_accuracy, alternative='greater')
            accuracy = n_correct/n_oof
            # relative error reduction, in [0,1]
            # e.g. (90% acc -> 99% acc) = 0.9, (50% acc -> 59% acc) = 0.18
            dist = 1 - (1 - accuracy)/(1-baseline_accuracy)
            dist = max(0, dist)  # below 0 = no evidence for drift
        else:
            probs_ref = probs_oof[y_oof == 0]
            probs_cur = probs_oof[y_oof == 1]
            dist, p_val = ks_2samp(probs_ref, probs_cur, alternative='greater')

        return p_val, dist

    @abstractmethod
    def score(self, x: np.ndarray) -> Tuple[float, float]:
        pass

    def predict(self, x: np.ndarray,  return_p_val: bool = True,
                return_distance: bool = True) -> Dict[Dict[str, str], Dict[str, Union[int, float]]]:
        """
        Predict whether a batch of data has drifted from the reference data.

        Parameters
        ----------
        x
            Batch of instances.
        return_p_val
            Whether to return the p-value of the test.
        return_distance
            Whether to return a notion of strength of the drift.
            K-S test stat if binarize_preds=False, otherwise relative error reduction.

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the drift prediction and optionally the performance of the classifier
            relative to its expectation under the no-change null.
        """
        # compute drift scores
        p_val, dist = self.score(x)
        drift_pred = int(p_val < self.p_val)

        # update reference dataset
        if isinstance(self.update_x_ref, dict) and self.preprocess_fn is not None and self.preprocess_x_ref:
            x = self.preprocess_fn(x)
        self.x_ref = update_reference(self.x_ref, x, self.n, self.update_x_ref)
        # used for reservoir sampling
        self.n += x.shape[0]  # type: ignore

        # populate drift dict
        # TODO: add instance level feedback
        cd = concept_drift_dict()
        cd['meta'] = self.meta
        cd['data']['is_drift'] = drift_pred
        if return_p_val:
            cd['data']['p_val'] = p_val
            cd['data']['threshold'] = self.p_val
        if return_distance:
            cd['data']['distance'] = dist
        return cd


class BaseMMDDrift(BaseDetector):
    def __init__(
            self,
            x_ref: np.ndarray,
            p_val: float = .05,
            preprocess_x_ref: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            sigma: Optional[np.ndarray] = None,
            configure_kernel_from_x_ref: bool = True,
            n_permutations: int = 100,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Maximum Mean Discrepancy (MMD) base data drift detector using a permutation test.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        p_val
            p-value used for the significance of the permutation test.
        preprocess_x_ref
            Whether to already preprocess and store the reference data.
        update_x_ref
            Reference data can optionally be updated to the last n instances seen by the detector
            or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while
            for reservoir sampling {'reservoir_sampling': n} is passed.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        sigma
            Optionally set the Gaussian RBF kernel bandwidth. Can also pass multiple bandwidth values as an array.
            The kernel evaluation is then averaged over those bandwidths.
        configure_kernel_from_x_ref
            Whether to already configure the kernel bandwidth from the reference data.
        n_permutations
            Number of permutations used in the permutation test.
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()

        if p_val is None:
            logger.warning('No p-value set for the drift threshold. Need to set it to detect data drift.')

        self.infer_sigma = configure_kernel_from_x_ref
        if configure_kernel_from_x_ref and isinstance(sigma, np.ndarray):
            self.infer_sigma = False
            logger.warning('`sigma` is specified for the kernel and `configure_kernel_from_x_ref` '
                           'is set to True. `sigma` argument takes priority over '
                           '`configure_kernel_from_x_ref` (set to False).')

        # optionally already preprocess reference data
        self.p_val = p_val
        if preprocess_x_ref and isinstance(preprocess_fn, Callable):  # type: ignore
            self.x_ref = preprocess_fn(x_ref)
        else:
            self.x_ref = x_ref
        self.preprocess_x_ref = preprocess_x_ref
        self.update_x_ref = update_x_ref
        self.preprocess_fn = preprocess_fn
        self.n = x_ref.shape[0]  # type: ignore
        self.n_permutations = n_permutations  # nb of iterations through permutation test

        # store input shape for save and load functionality
        self.input_shape = input_shape if isinstance(input_shape, tuple) else x_ref.shape[1:]

        # set metadata
        self.meta.update({'detector_type': 'offline', 'data_type': data_type})

    def preprocess(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Data preprocessing before computing the drift scores.
        Parameters
        ----------
        x
            Batch of instances.
        Returns
        -------
        Preprocessed reference data and new instances.
        """
        if isinstance(self.preprocess_fn, Callable):  # type: ignore
            x = self.preprocess_fn(x)
            x_ref = self.x_ref if self.preprocess_x_ref else self.preprocess_fn(self.x_ref)
            return x_ref, x
        else:
            return self.x_ref, x

    @abstractmethod
    def kernel_matrix(self, x: Union['torch.Tensor', 'tf.Tensor'], y: Union['torch.Tensor', 'tf.Tensor']) \
            -> Union['torch.Tensor', 'tf.Tensor']:
        pass

    @abstractmethod
    def score(self, x: np.ndarray) -> Tuple[float, float, np.ndarray]:
        pass

    def predict(self, x: np.ndarray, return_p_val: bool = True, return_distance: bool = True) \
            -> Dict[Dict[str, str], Dict[str, Union[int, float]]]:
        """
        Predict whether a batch of data has drifted from the reference data.

        Parameters
        ----------
        x
            Batch of instances.
        return_p_val
            Whether to return the p-value of the permutation test.
        return_distance
            Whether to return the MMD metric between the new batch and reference data.

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the drift prediction and optionally the p-value, threshold and MMD metric.
        """
        # compute drift scores
        p_val, dist, dist_permutations = self.score(x)
        drift_pred = int(p_val < self.p_val)

        # compute distance threshold
        idx_threshold = int(self.p_val * len(dist_permutations))
        distance_threshold = np.sort(dist_permutations)[::-1][idx_threshold]

        # update reference dataset
        if isinstance(self.update_x_ref, dict) and self.preprocess_fn is not None and self.preprocess_x_ref:
            x = self.preprocess_fn(x)
        self.x_ref = update_reference(self.x_ref, x, self.n, self.update_x_ref)
        # used for reservoir sampling
        self.n += x.shape[0]  # type: ignore

        # populate drift dict
        cd = concept_drift_dict()
        cd['meta'] = self.meta
        cd['data']['is_drift'] = drift_pred
        if return_p_val:
            cd['data']['p_val'] = p_val
            cd['data']['threshold'] = self.p_val
        if return_distance:
            cd['data']['distance'] = dist
            cd['data']['distance_threshold'] = distance_threshold
        return cd


class BaseUnivariateDrift(BaseDetector):
    def __init__(
            self,
            x_ref: np.ndarray,
            p_val: float = .05,
            preprocess_x_ref: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            correction: str = 'bonferroni',
            n_features: Optional[int] = None,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Generic drift detector component which serves as a base class for methods using
        univariate tests with multivariate correction.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        p_val
            p-value used for significance of the statistical test for each feature. If the FDR correction method
            is used, this corresponds to the acceptable q-value.
        preprocess_x_ref
            Whether to already preprocess and store the reference data.
        update_x_ref
            Reference data can optionally be updated to the last n instances seen by the detector
            or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while
            for reservoir sampling {'reservoir_sampling': n} is passed.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
            Typically a dimensionality reduction technique.
        correction
            Correction type for multivariate data. Either 'bonferroni' or 'fdr' (False Discovery Rate).
        n_features
            Number of features used in the statistical test. No need to pass it if no preprocessing takes place.
            In case of a preprocessing step, this can also be inferred automatically but could be more
            expensive to compute.
        input_shape
            Shape of input data. Needs to be provided for text data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()

        if p_val is None:
            logger.warning('No p-value set for the drift threshold. Need to set it to detect data drift.')

        # optionally already preprocess reference data
        self.p_val = p_val
        if preprocess_x_ref and isinstance(preprocess_fn, Callable):  # type: ignore
            self.x_ref = preprocess_fn(x_ref)
        else:
            self.x_ref = x_ref
        self.preprocess_x_ref = preprocess_x_ref
        self.update_x_ref = update_x_ref
        self.preprocess_fn = preprocess_fn
        self.correction = correction
        self.n = x_ref.shape[0]  # type: ignore

        # store input shape for save and load functionality
        self.input_shape = input_shape if isinstance(input_shape, tuple) else x_ref.shape[1:]

        # compute number of features for the univariate tests
        if isinstance(n_features, int):
            self.n_features = n_features
        elif not isinstance(preprocess_fn, Callable) or preprocess_x_ref:
            # infer features from preprocessed reference data
            self.n_features = self.x_ref.reshape(self.x_ref.shape[0], -1).shape[-1]
        else:  # infer number of features after applying preprocessing step
            x = self.preprocess_fn(x_ref[0:1])
            self.n_features = x.reshape(x.shape[0], -1).shape[-1]

        if correction not in ['bonferroni', 'fdr'] and self.n_features > 1:
            raise ValueError('Only `bonferroni` and `fdr` are acceptable for multivariate correction.')

        # set metadata
        self.meta['detector_type'] = 'offline'  # offline refers to fitting the CDF for K-S
        self.meta['data_type'] = data_type

    def preprocess(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Data preprocessing before computing the drift scores.

        Parameters
        ----------
        x
            Batch of instances.

        Returns
        -------
        Preprocessed reference data and new instances.
        """
        if isinstance(self.preprocess_fn, Callable):  # type: ignore
            x = self.preprocess_fn(x)
            x_ref = self.x_ref if self.preprocess_x_ref else self.preprocess_fn(self.x_ref)
            return x_ref, x
        else:
            return self.x_ref, x

    @abstractmethod
    def feature_score(self, x_ref: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def score(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the feature-wise drift score which is the p-value of the
        statistical test and the test statistic.

        Parameters
        ----------
        x
            Batch of instances.

        Returns
        -------
        Feature level p-values and test statistics.
        """
        x_ref, x = self.preprocess(x)
        score, dist = self.feature_score(x_ref, x)  # feature-wise univariate test
        return score, dist

    def predict(self, x: np.ndarray, drift_type: str = 'batch',
                return_p_val: bool = True, return_distance: bool = True) \
            -> Dict[Dict[str, str], Dict[str, Union[np.ndarray, int, float]]]:
        """
        Predict whether a batch of data has drifted from the reference data.

        Parameters
        ----------
        x
            Batch of instances.
        drift_type
            Predict drift at the 'feature' or 'batch' level. For 'batch', the test statistics for
            each feature are aggregated using the Bonferroni or False Discovery Rate correction.
        return_p_val
            Whether to return feature level p-values.
        return_distance
            Whether to return the test statistic between the features of the new batch and reference data.

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the drift prediction and optionally the feature level p-values,
         threshold after multivariate correction if needed and test statistics.
        """
        # compute drift scores
        p_vals, dist = self.score(x)

        # TODO: return both feature-level and batch-level drift predictions by default
        # values below p-value threshold are drift
        if drift_type == 'feature':
            drift_pred = (p_vals < self.p_val).astype(int)
        elif drift_type == 'batch' and self.correction == 'bonferroni':
            threshold = self.p_val / self.n_features
            drift_pred = int((p_vals < threshold).any())
        elif drift_type == 'batch' and self.correction == 'fdr':
            drift_pred, threshold = fdr(p_vals, q_val=self.p_val)
        else:
            raise ValueError('`drift_type` needs to be either `feature` or `batch`.')

        # update reference dataset
        if isinstance(self.update_x_ref, dict) and self.preprocess_fn is not None and self.preprocess_x_ref:
            x = self.preprocess_fn(x)
        self.x_ref = update_reference(self.x_ref, x, self.n, self.update_x_ref)
        # used for reservoir sampling
        self.n += x.shape[0]  # type: ignore

        # populate drift dict
        cd = concept_drift_dict()
        cd['meta'] = self.meta
        cd['data']['is_drift'] = drift_pred
        if return_p_val:
            cd['data']['p_val'] = p_vals
            cd['data']['threshold'] = self.p_val if drift_type == 'feature' else threshold
        if return_distance:
            cd['data']['distance'] = dist
        return cd
