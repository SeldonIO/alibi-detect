import logging
from abc import abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

import numpy as np
from alibi_detect.base import BaseDetector, concept_drift_dict, DriftConfigMixin
from alibi_detect.cd.utils import get_input_shape, update_reference
from alibi_detect.utils.frameworks import has_pytorch, has_tensorflow
from alibi_detect.utils.statstest import fdr
from scipy.stats import binom_test, ks_2samp
from sklearn.model_selection import StratifiedKFold

if has_pytorch:
    import torch

if has_tensorflow:
    import tensorflow as tf

logger = logging.getLogger(__name__)


class BaseClassifierDrift(BaseDetector):
    model: Union['tf.keras.Model', 'torch.nn.Module']

    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            p_val: float = .05,
            x_ref_preprocessed: bool = False,
            preprocess_at_init: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            preds_type: str = 'probs',
            binarize_preds: bool = False,
            train_size: Optional[float] = .75,
            n_folds: Optional[int] = None,
            retrain_from_scratch: bool = True,
            seed: int = 0,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None,
    ) -> None:
        """
        A context-aware drift detector based on a conditional analogue of the maximum mean discrepancy (MMD).
        Only detects differences between samples that can not be attributed to differences between associated
        sets of contexts. p-values are computed using a conditional permutation test.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        p_val
            p-value used for the significance of the test.
        x_ref_preprocessed
            Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only
            the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference
            data will also be preprocessed.
        preprocess_at_init
            Whether to preprocess the reference data when the detector is instantiated. Otherwise, the reference
            data will be preprocessed at prediction time. Only applies if `x_ref_preprocessed=False`.
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
        retrain_from_scratch
            Whether the classifier should be retrained from scratch for each set of test data or whether
            it should instead continue training from where it left off on the previous set.
        seed
            Optional random seed for fold selection.
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()

        if p_val is None:
            logger.warning('No p-value set for the drift threshold. Need to set it to detect data drift.')

        if isinstance(train_size, float) and isinstance(n_folds, int):
            logger.warning('Both `n_folds` and `train_size` specified. By default `n_folds` is used.')

        if n_folds is not None and n_folds > 1 and not retrain_from_scratch:
            raise ValueError("If using multiple folds the model must be retrained from scratch for each fold.")

        # x_ref preprocessing
        self.preprocess_at_init = preprocess_at_init
        self.x_ref_preprocessed = x_ref_preprocessed
        if preprocess_fn is not None and not isinstance(preprocess_fn, Callable):  # type: ignore[arg-type]
            raise ValueError("`preprocess_fn` is not a valid Callable.")
        if self.preprocess_at_init and not self.x_ref_preprocessed and preprocess_fn is not None:
            self.x_ref = preprocess_fn(x_ref)
        else:
            self.x_ref = x_ref

        # Other attributes
        self.p_val = p_val
        self.update_x_ref = update_x_ref
        self.preprocess_fn = preprocess_fn
        self.n = len(x_ref)

        # define whether soft preds and optionally the stratified k-fold split
        self.preds_type = preds_type
        self.binarize_preds = binarize_preds
        if isinstance(n_folds, int):
            self.train_size = None
            self.skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        else:
            self.train_size, self.skf = train_size, None
        self.retrain_from_scratch = retrain_from_scratch

        # store input shape for save and load functionality
        self.input_shape = get_input_shape(input_shape, x_ref)

        # set metadata
        self.meta['online'] = False
        self.meta['data_type'] = data_type
        self.meta['detector_type'] = 'drift'
        self.meta['params'] = {'binarize_preds ': binarize_preds, 'preds_type': preds_type}

    def preprocess(self, x: Union[np.ndarray, list]) -> Tuple[Union[np.ndarray, list], Union[np.ndarray, list]]:
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
        if self.preprocess_fn is not None:
            x = self.preprocess_fn(x)
            if not self.preprocess_at_init and not self.x_ref_preprocessed:
                x_ref = self.preprocess_fn(self.x_ref)
            else:
                x_ref = self.x_ref
            return x_ref, x  # type: ignore[return-value]
        else:
            return self.x_ref, x  # type: ignore[return-value]

    def get_splits(
            self,
            x_ref: Union[np.ndarray, list],
            x: Union[np.ndarray, list],
            return_splits: bool = True
    ) -> Union[Tuple[Union[np.ndarray, list], np.ndarray],
               Tuple[Union[np.ndarray, list], np.ndarray, Optional[List[Tuple[np.ndarray, np.ndarray]]]]]:
        """
        Split reference and test data in train and test folds used by the classifier.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        x
            Batch of instances.
        return_splits
            Whether to return the splits.

        Returns
        -------
        Combined reference and test instances with labels and optionally a list with tuples of
        train and test indices for optionally different folds.
        """
        # create dataset and labels
        y = np.concatenate([np.zeros(len(x_ref)), np.ones(len(x))], axis=0).astype(np.int64)  # Fix #411
        if isinstance(x_ref, np.ndarray) and isinstance(x, np.ndarray):
            x = np.concatenate([x_ref, x], axis=0)
        else:  # add 2 lists
            x = x_ref + x

        if not return_splits:
            return x, y

        # random shuffle if stratified folds are not used
        n_tot = len(x)
        if self.skf is None:
            idx_shuffle = np.random.choice(np.arange(n_tot), size=n_tot, replace=False)
            n_tr = int(n_tot * self.train_size)
            idx_tr, idx_te = idx_shuffle[:n_tr], idx_shuffle[n_tr:]
            splits = [(idx_tr, idx_te)]
        else:  # use stratified folds
            splits = self.skf.split(np.zeros(n_tot), y)
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
            Size of current window used in training model

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
            accuracy = n_correct / n_oof
            # relative error reduction, in [0,1]
            # e.g. (90% acc -> 99% acc) = 0.9, (50% acc -> 59% acc) = 0.18
            dist = 1 - (1 - accuracy) / (1 - baseline_accuracy)
            dist = max(0, dist)  # below 0 = no evidence for drift
        else:
            probs_ref = probs_oof[y_oof == 0]
            probs_cur = probs_oof[y_oof == 1]
            dist, p_val = ks_2samp(probs_ref, probs_cur, alternative='greater')

        return p_val, dist

    @abstractmethod
    def score(self, x: Union[np.ndarray, list]) -> Tuple[float, float, np.ndarray, np.ndarray]:
        pass

    def predict(self, x: Union[np.ndarray, list], return_p_val: bool = True,
                return_distance: bool = True, return_probs: bool = True, return_model: bool = True) \
            -> Dict[str, Dict[str, Union[str, int, float, Callable]]]:
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
        return_probs
            Whether to return the instance level classifier probabilities for the reference and test data
            (0=reference data, 1=test data).
        return_model
            Whether to return the updated model trained to discriminate reference and test instances.

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the drift prediction and optionally the p-value, performance of the classifier
        relative to its expectation under the no-change null, the out-of-fold classifier model
        prediction probabilities on the reference and test data, and the trained model.
        """
        # compute drift scores
        p_val, dist, probs_ref, probs_test = self.score(x)
        drift_pred = int(p_val < self.p_val)

        # update reference dataset
        if isinstance(self.update_x_ref, dict) and self.preprocess_fn is not None and self.preprocess_at_init:
            x = self.preprocess_fn(x)
        # TODO: TBD: can `x` ever be a `list` after pre-processing? update_references and downstream functions
        # don't support list inputs and without the type: ignore[arg-type] mypy complains
        self.x_ref = update_reference(self.x_ref, x, self.n, self.update_x_ref)  # type: ignore[arg-type]
        # used for reservoir sampling
        self.n += len(x)

        # populate drift dict
        cd = concept_drift_dict()
        cd['meta'] = self.meta
        cd['data']['is_drift'] = drift_pred
        if return_p_val:
            cd['data']['p_val'] = p_val
            cd['data']['threshold'] = self.p_val
        if return_distance:
            cd['data']['distance'] = dist
        if return_probs:
            cd['data']['probs_ref'] = probs_ref
            cd['data']['probs_test'] = probs_test
        if return_model:
            cd['data']['model'] = self.model
        return cd


class BaseLearnedKernelDrift(BaseDetector):
    kernel: Union['tf.keras.Model', 'torch.nn.Module']

    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            p_val: float = .05,
            x_ref_preprocessed: bool = False,
            preprocess_at_init: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            n_permutations: int = 100,
            train_size: Optional[float] = .75,
            retrain_from_scratch: bool = True,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Base class for the learned kernel-based drift detector.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        p_val
            p-value used for the significance of the test.
        x_ref_preprocessed
            Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only
            the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference
            data will also be preprocessed.
        preprocess_at_init
            Whether to preprocess the reference data when the detector is instantiated. Otherwise, the reference
            data will be preprocessed at prediction time. Only applies if `x_ref_preprocessed=False`.
        update_x_ref
            Reference data can optionally be updated to the last n instances seen by the detector
            or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while
            for reservoir sampling {'reservoir_sampling': n} is passed.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        n_permutations
            The number of permutations to use in the permutation test once the MMD has been computed.
        train_size
            Optional fraction (float between 0 and 1) of the dataset used to train the kernel.
            The drift is detected on `1 - train_size`. Cannot be used in combination with `n_folds`.
        retrain_from_scratch
            Whether the kernel should be retrained from scratch for each set of test data or whether
            it should instead continue training from where it left off on the previous set.
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()
        if p_val is None:
            logger.warning('No p-value set for the drift threshold. Need to set it to detect data drift.')

        # x_ref preprocessing
        self.preprocess_at_init = preprocess_at_init
        self.x_ref_preprocessed = x_ref_preprocessed
        if preprocess_fn is not None and not isinstance(preprocess_fn, Callable):  # type: ignore[arg-type]
            raise ValueError("`preprocess_fn` is not a valid Callable.")
        if self.preprocess_at_init and not self.x_ref_preprocessed and preprocess_fn is not None:
            self.x_ref = preprocess_fn(x_ref)
        else:
            self.x_ref = x_ref

        # Other attributes
        self.p_val = p_val
        self.update_x_ref = update_x_ref
        self.preprocess_fn = preprocess_fn
        self.n = len(x_ref)

        self.n_permutations = n_permutations
        self.train_size = train_size
        self.retrain_from_scratch = retrain_from_scratch

        # store input shape for save and load functionality
        self.input_shape = get_input_shape(input_shape, x_ref)

        # set metadata
        self.meta['detector_type'] = 'drift'
        self.meta['data_type'] = data_type
        self.meta['online'] = False

    def preprocess(self, x: Union[np.ndarray, list]) -> Tuple[Union[np.ndarray, list], Union[np.ndarray, list]]:
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
        if self.preprocess_fn is not None:
            x = self.preprocess_fn(x)
            if not self.preprocess_at_init and not self.x_ref_preprocessed:
                x_ref = self.preprocess_fn(self.x_ref)
            else:
                x_ref = self.x_ref
            return x_ref, x  # type: ignore[return-value]
        else:
            return self.x_ref, x  # type: ignore[return-value]

    def get_splits(self, x_ref: Union[np.ndarray, list], x: Union[np.ndarray, list]) \
            -> Tuple[Tuple[Union[np.ndarray, list], Union[np.ndarray, list]],
                     Tuple[Union[np.ndarray, list], Union[np.ndarray, list]]]:
        """
        Split reference and test data into two splits -- one of which to learn test locations
        and parameters and one to use for tests.
        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        x
            Batch of instances.
        Returns
        -------
        Tuple containing split train data and tuple containing split test data
        """

        n_ref, n_cur = len(x_ref), len(x)
        perm_ref, perm_cur = np.random.permutation(n_ref), np.random.permutation(n_cur)
        idx_ref_tr, idx_ref_te = perm_ref[:int(n_ref * self.train_size)], perm_ref[int(n_ref * self.train_size):]
        idx_cur_tr, idx_cur_te = perm_cur[:int(n_cur * self.train_size)], perm_cur[int(n_cur * self.train_size):]

        if isinstance(x_ref, np.ndarray):
            x_ref_tr, x_ref_te = x_ref[idx_ref_tr], x_ref[idx_ref_te]
            x_cur_tr, x_cur_te = x[idx_cur_tr], x[idx_cur_te]  # type: ignore[call-overload]
        elif isinstance(x, list):
            x_ref_tr, x_ref_te = [x_ref[_] for _ in idx_ref_tr], [x_ref[_] for _ in idx_ref_te]
            x_cur_tr, x_cur_te = [x[_] for _ in idx_cur_tr], [x[_] for _ in idx_cur_te]
        else:
            raise TypeError(f'x needs to be of type np.ndarray or list and not {type(x)}.')

        return (x_ref_tr, x_cur_tr), (x_ref_te, x_cur_te)

    @abstractmethod
    def score(self, x: Union[np.ndarray, list]) -> Tuple[float, float, float]:
        pass

    def predict(self, x: Union[np.ndarray, list], return_p_val: bool = True,
                return_distance: bool = True, return_kernel: bool = True) \
            -> Dict[Dict[str, str], Dict[str, Union[int, float, Callable]]]:
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
        return_kernel
            Whether to return the updated kernel trained to discriminate reference and test instances.

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the detector's metadata.
        'data' contains the drift prediction and optionally the p-value, threshold, MMD metric and
            trained kernel.
        """
        # compute drift scores
        p_val, dist, distance_threshold = self.score(x)
        drift_pred = int(p_val < self.p_val)

        # update reference dataset
        if isinstance(self.update_x_ref, dict) and self.preprocess_fn is not None and self.preprocess_at_init:
            x = self.preprocess_fn(x)
        self.x_ref = update_reference(self.x_ref, x, self.n, self.update_x_ref)  # type: ignore[arg-type]
        # used for reservoir sampling
        self.n += len(x)

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
        if return_kernel:
            cd['data']['kernel'] = self.kernel
        return cd


class BaseMMDDrift(BaseDetector):
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            p_val: float = .05,
            x_ref_preprocessed: bool = False,
            preprocess_at_init: bool = True,
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
        x_ref_preprocessed
            Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only
            the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference
            data will also be preprocessed.
        preprocess_at_init
            Whether to preprocess the reference data when the detector is instantiated. Otherwise, the reference
            data will be preprocessed at prediction time. Only applies if `x_ref_preprocessed=False`.
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

        # x_ref preprocessing
        self.preprocess_at_init = preprocess_at_init
        self.x_ref_preprocessed = x_ref_preprocessed
        if preprocess_fn is not None and not isinstance(preprocess_fn, Callable):  # type: ignore[arg-type]
            raise ValueError("`preprocess_fn` is not a valid Callable.")
        if self.preprocess_at_init and not self.x_ref_preprocessed and preprocess_fn is not None:
            self.x_ref = preprocess_fn(x_ref)
        else:
            self.x_ref = x_ref

        # Other attributes
        self.p_val = p_val
        self.update_x_ref = update_x_ref
        self.preprocess_fn = preprocess_fn
        self.n = len(x_ref)
        self.n_permutations = n_permutations  # nb of iterations through permutation test

        # store input shape for save and load functionality
        self.input_shape = get_input_shape(input_shape, x_ref)

        # set metadata
        self.meta.update({'detector_type': 'drift', 'online': False, 'data_type': data_type})

    def preprocess(self, x: Union[np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray]:
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
        if self.preprocess_fn is not None:
            x = self.preprocess_fn(x)
            if not self.preprocess_at_init and not self.x_ref_preprocessed:
                x_ref = self.preprocess_fn(self.x_ref)
                # TODO: TBD: similar to above, can x be a list here? x_ref is also revealed to be Any,
                #  can we tighten the type up (e.g. by typing Callable with stricter inputs/outputs?
            else:
                x_ref = self.x_ref
            return x_ref, x  # type: ignore[return-value]
        else:
            return self.x_ref, x  # type: ignore[return-value]

    @abstractmethod
    def kernel_matrix(self, x: Union['torch.Tensor', 'tf.Tensor'], y: Union['torch.Tensor', 'tf.Tensor']) \
            -> Union['torch.Tensor', 'tf.Tensor']:
        pass

    @abstractmethod
    def score(self, x: Union[np.ndarray, list]) -> Tuple[float, float, float]:
        pass

    def predict(self, x: Union[np.ndarray, list], return_p_val: bool = True, return_distance: bool = True) \
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
        p_val, dist, distance_threshold = self.score(x)
        drift_pred = int(p_val < self.p_val)

        # update reference dataset
        if isinstance(self.update_x_ref, dict) and self.preprocess_fn is not None and self.preprocess_at_init:
            x = self.preprocess_fn(x)
        self.x_ref = update_reference(self.x_ref, x, self.n, self.update_x_ref)  # type: ignore[arg-type]
        # used for reservoir sampling
        self.n += len(x)

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


class BaseLSDDDrift(BaseDetector):
    # TODO: TBD: this is only created when _configure_normalization is called from backend-specific classes,
    # is declaring it here the right thing to do?
    _normalize: Callable
    _unnormalize: Callable

    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            p_val: float = .05,
            x_ref_preprocessed: bool = False,
            preprocess_at_init: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            sigma: Optional[np.ndarray] = None,
            n_permutations: int = 100,
            n_kernel_centers: Optional[int] = None,
            lambda_rd_max: float = 0.2,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Least-squares Density Difference (LSDD) base data drift detector using a permutation test.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        p_val
            p-value used for the significance of the permutation test.
        x_ref_preprocessed
            Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only
            the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference
            data will also be preprocessed.
        preprocess_at_init
            Whether to preprocess the reference data when the detector is instantiated. Otherwise, the reference
            data will be preprocessed at prediction time. Only applies if `x_ref_preprocessed=False`.
        update_x_ref
            Reference data can optionally be updated to the last n instances seen by the detector
            or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while
            for reservoir sampling {'reservoir_sampling': n} is passed.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        sigma
            Optionally set the bandwidth of the Gaussian kernel used in estimating the LSDD. Can also pass multiple
            bandwidth values as an array. The kernel evaluation is then averaged over those bandwidths. If `sigma`
            is not specified, the 'median heuristic' is adopted whereby `sigma` is set as the median pairwise distance
            between reference samples.
        n_permutations
            Number of permutations used in the permutation test.
        n_kernel_centers
            The number of reference samples to use as centers in the Gaussian kernel model used to estimate LSDD.
            Defaults to 1/20th of the reference data.
        lambda_rd_max
            The maximum relative difference between two estimates of LSDD that the regularization parameter
            lambda is allowed to cause. Defaults to 0.2 as in the paper.
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()
        if p_val is None:
            logger.warning('No p-value set for the drift threshold. Need to set it to detect data drift.')

        # x_ref preprocessing
        self.preprocess_at_init = preprocess_at_init
        self.x_ref_preprocessed = x_ref_preprocessed
        if preprocess_fn is not None and not isinstance(preprocess_fn, Callable):  # type: ignore[arg-type]
            raise ValueError("`preprocess_fn` is not a valid Callable.")
        if self.preprocess_at_init and not self.x_ref_preprocessed and preprocess_fn is not None:
            self.x_ref = preprocess_fn(x_ref)
        else:
            self.x_ref = x_ref

        # Other attributes
        self.p_val = p_val
        self.sigma = sigma
        self.update_x_ref = update_x_ref
        self.preprocess_fn = preprocess_fn
        self.n = len(x_ref)
        self.n_permutations = n_permutations  # nb of iterations through permutation test
        self.n_kernel_centers = n_kernel_centers or max(self.n // 20, 1)
        self.lambda_rd_max = lambda_rd_max

        # store input shape for save and load functionality
        self.input_shape = get_input_shape(input_shape, x_ref)

        # set metadata
        self.meta.update({'detector_type': 'drift', 'online': False, 'data_type': data_type})

    def preprocess(self, x: Union[np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray]:
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
        if self.preprocess_fn is not None:
            x = self.preprocess_fn(x)
            if not self.preprocess_at_init and not self.x_ref_preprocessed:
                x_ref = self.preprocess_fn(self.x_ref)
            else:
                x_ref = self.x_ref
            return x_ref, x  # type: ignore[return-value]
        else:
            return self.x_ref, x  # type: ignore[return-value]

    @abstractmethod
    def score(self, x: Union[np.ndarray, list]) -> Tuple[float, float, float]:
        pass

    def predict(self, x: Union[np.ndarray, list], return_p_val: bool = True, return_distance: bool = True) \
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
            Whether to return the LSDD metric between the new batch and reference data.

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the drift prediction and optionally the p-value, threshold and LSDD metric.
        """
        # compute drift scores
        p_val, dist, distance_threshold = self.score(x)
        drift_pred = int(p_val < self.p_val)

        # update reference dataset
        if isinstance(self.update_x_ref, dict):
            if self.preprocess_fn is not None and self.preprocess_at_init:
                x = self.preprocess_fn(x)
                x = self._normalize(x)
            elif self.preprocess_fn is None:
                x = self._normalize(x)
            else:
                pass
        self.x_ref = update_reference(self.x_ref, x, self.n, self.update_x_ref)  # type: ignore[arg-type]
        # used for reservoir sampling
        self.n += len(x)  # type: ignore

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


class BaseUnivariateDrift(BaseDetector, DriftConfigMixin):
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            p_val: float = .05,
            x_ref_preprocessed: bool = False,
            preprocess_at_init: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            correction: str = 'bonferroni',
            n_features: Optional[int] = None,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Generic drift detector component which serves as a base class for methods using
        univariate tests. If n_features > 1, a multivariate correction is applied such that
        the false positive rate is upper bounded by the specified p-value, with equality in
        the case of independent features.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        p_val
            p-value used for significance of the statistical test for each feature. If the FDR correction method
            is used, this corresponds to the acceptable q-value.
        x_ref_preprocessed
            Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only
            the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference
            data will also be preprocessed.
        preprocess_at_init
            Whether to preprocess the reference data when the detector is instantiated. Otherwise, the reference
            data will be preprocessed at prediction time. Only applies if `x_ref_preprocessed=False`.
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

        # x_ref preprocessing
        self.preprocess_at_init = preprocess_at_init
        self.x_ref_preprocessed = x_ref_preprocessed
        if preprocess_fn is not None and not isinstance(preprocess_fn, Callable):  # type: ignore[arg-type]
            raise ValueError("`preprocess_fn` is not a valid Callable.")
        if self.preprocess_at_init and not self.x_ref_preprocessed and preprocess_fn is not None:
            self.x_ref = preprocess_fn(x_ref)
        else:
            self.x_ref = x_ref

        # Other attributes
        self.p_val = p_val
        self.update_x_ref = update_x_ref
        self.preprocess_fn = preprocess_fn
        self.correction = correction
        self.n = len(x_ref)

        # store input shape for save and load functionality
        self.input_shape = get_input_shape(input_shape, x_ref)

        # compute number of features for the univariate tests
        if isinstance(n_features, int):
            self.n_features = n_features
        elif not isinstance(preprocess_fn, Callable) or preprocess_at_init or x_ref_preprocessed:
            # infer features from preprocessed reference data
            self.n_features = self.x_ref.reshape(self.x_ref.shape[0], -1).shape[-1]
        else:  # infer number of features after applying preprocessing step
            x = self.preprocess_fn(x_ref[0:1])
            self.n_features = x.reshape(x.shape[0], -1).shape[-1]

        if correction not in ['bonferroni', 'fdr'] and self.n_features > 1:
            raise ValueError('Only `bonferroni` and `fdr` are acceptable for multivariate correction.')

        # set metadata
        self.meta['online'] = False  # offline refers to fitting the CDF for K-S
        self.meta['data_type'] = data_type
        self.meta['detector_type'] = 'drift'

    def preprocess(self, x: Union[np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray]:
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
        if self.preprocess_fn is not None:
            x = self.preprocess_fn(x)
            if not self.preprocess_at_init and not self.x_ref_preprocessed:
                x_ref = self.preprocess_fn(self.x_ref)
            else:
                x_ref = self.x_ref
            return x_ref, x  # type: ignore[return-value]
        else:
            return self.x_ref, x  # type: ignore[return-value]

    @abstractmethod
    def feature_score(self, x_ref: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def score(self, x: Union[np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray]:
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

    def predict(self, x: Union[np.ndarray, list], drift_type: str = 'batch',
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
            each feature are aggregated using the Bonferroni or False Discovery Rate correction (if n_features>1).
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
            drift_pred = int((p_vals < threshold).any())  # type: ignore[assignment]
        elif drift_type == 'batch' and self.correction == 'fdr':
            drift_pred, threshold = fdr(p_vals, q_val=self.p_val)  # type: ignore[assignment]
        else:
            raise ValueError('`drift_type` needs to be either `feature` or `batch`.')

        # update reference dataset
        if isinstance(self.update_x_ref, dict) and self.preprocess_fn is not None and self.preprocess_at_init:
            x = self.preprocess_fn(x)
        self.x_ref = update_reference(self.x_ref, x, self.n, self.update_x_ref)  # type: ignore[arg-type]
        # used for reservoir sampling
        self.n += len(x)

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


class BaseContextMMDDrift(BaseDetector):
    lams: Optional[Tuple[Any, Any]] = None

    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            c_ref: np.ndarray,
            p_val: float = .05,
            x_ref_preprocessed: bool = False,
            preprocess_at_init: bool = True,
            update_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            x_kernel: Callable = None,
            c_kernel: Callable = None,
            n_permutations: int = 1000,
            prop_c_held: float = 0.25,
            n_folds: int = 5,
            batch_size: Optional[int] = 256,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None,
            verbose: bool = False,
    ) -> None:
        """
        Maximum Mean Discrepancy (MMD) based context aware drift detector.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        c_ref
            Context for the reference distribution.
        p_val
            p-value used for the significance of the permutation test.
        x_ref_preprocessed
            Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only
            the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference
            data will also be preprocessed.
        preprocess_at_init
            Whether to preprocess the reference data when the detector is instantiated. Otherwise, the reference
            data will be preprocessed at prediction time. Only applies if `x_ref_preprocessed=False`.
        update_ref
            Reference data can optionally be updated to the last N instances seen by the detector.
            The parameter should be passed as a dictionary *{'last': N}*.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        x_kernel
            Kernel defined on the input data, defaults to Gaussian RBF kernel.
        c_kernel
            Kernel defined on the context data, defaults to Gaussian RBF kernel.
        n_permutations
            Number of permutations used in the permutation test.
        prop_c_held
            Proportion of contexts held out to condition on.
        n_folds
            Number of cross-validation folds used when tuning the regularisation parameters.
        batch_size
            If not None, then compute batches of MMDs at a time (rather than all at once).
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        verbose
            Whether or not to print progress during configuration.
        """
        super().__init__()
        if p_val is None:
            logger.warning('No p-value set for the drift threshold. Need to set it to detect data drift.')

        # x_ref preprocessing
        self.preprocess_at_init = preprocess_at_init
        self.x_ref_preprocessed = x_ref_preprocessed
        if preprocess_fn is not None and not isinstance(preprocess_fn, Callable):  # type: ignore[arg-type]
            raise ValueError("`preprocess_fn` is not a valid Callable.")
        if self.preprocess_at_init and not self.x_ref_preprocessed and preprocess_fn is not None:
            self.x_ref = preprocess_fn(x_ref)
        else:
            self.x_ref = x_ref

        # Other attributes
        self.p_val = p_val
        self.preprocess_fn = preprocess_fn
        self.n = len(x_ref)
        self.n_permutations = n_permutations  # nb of iterations through permutation test
        self.x_kernel = x_kernel
        self.c_kernel = c_kernel
        if len(c_ref) == self.n:
            self.c_ref = c_ref
        else:
            raise ValueError('x_ref and c_ref should contain the same number of instances.')

        # store input shape for save and load functionality
        self.input_shape = get_input_shape(input_shape, x_ref)

        # Regularisation parameter tuning settings
        if n_folds > 1:
            self.n_folds = n_folds
        else:
            raise ValueError('The `n_folds` parameter must be > 1.')
        self.lams = None

        # Update ref attribute. Disallow res
        self.update_ref = update_ref
        if update_ref is not None:
            if 'reservoir_sampling' in update_ref.keys():
                raise ValueError("The BaseContextMMDDrift detector doesn't currently support the `reservoir_sampling` "
                                 "option in `update_ref`.")

        # Other attributes
        self.prop_c_held = prop_c_held
        self.batch_size = batch_size
        self.verbose = verbose

        # set metadata
        self.meta.update({'detector_type': 'drift', 'online': False, 'data_type': data_type})

    def preprocess(self, x: Union[np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray]:
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
        if self.preprocess_fn is not None:
            x = self.preprocess_fn(x)
            if not self.preprocess_at_init and not self.x_ref_preprocessed:
                x_ref = self.preprocess_fn(self.x_ref)
            else:
                x_ref = self.x_ref
            return x_ref, x  # type: ignore[return-value]
        else:
            return self.x_ref, x  # type: ignore[return-value]

    @abstractmethod
    def score(self,  # type: ignore[override]
              x: Union[np.ndarray, list], c: np.ndarray) -> Tuple[float, float, float, Tuple]:
        pass

    def predict(self,  # type: ignore[override]
                x: Union[np.ndarray, list], c: np.ndarray,
                return_p_val: bool = True, return_distance: bool = True, return_coupling: bool = False) \
            -> Dict[Dict[str, str], Dict[str, Union[int, float]]]:
        """
        Predict whether a batch of data has drifted from the reference data, given the provided context.

        Parameters
        ----------
        x
            Batch of instances.
        c
            Context associated with batch of instances.
        return_p_val
            Whether to return the p-value of the permutation test.
        return_distance
            Whether to return the conditional MMD test statistic between the new batch and reference data.
        return_coupling
            Whether to return the coupling matrices.

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the drift prediction and optionally the p-value, threshold, conditional MMD test statistic
        and coupling matrices.
        """
        # compute drift scores
        p_val, dist, distance_threshold, coupling = self.score(x, c)
        drift_pred = int(p_val < self.p_val)

        # update reference dataset
        if isinstance(self.update_ref, dict) and self.preprocess_fn is not None and self.preprocess_at_init:
            x = self.preprocess_fn(x)
        self.x_ref = update_reference(self.x_ref, x, self.n, self.update_ref)  # type: ignore[arg-type]
        self.c_ref = update_reference(self.c_ref, c, self.n, self.update_ref)  # type: ignore[arg-type]
        # used for reservoir sampling
        self.n += len(x)

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
        if return_coupling:
            cd['data']['coupling_xx'] = coupling[0]
            cd['data']['coupling_yy'] = coupling[1]
            cd['data']['coupling_xy'] = coupling[2]
        return cd
