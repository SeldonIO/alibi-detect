import logging
import warnings
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, TYPE_CHECKING

import numpy as np
from alibi_detect.base import BaseDetector, concept_drift_dict
from alibi_detect.cd.utils import get_input_shape
from alibi_detect.utils.state import StateMixin
from alibi_detect.utils._types import Literal

if TYPE_CHECKING:
    import torch
    import tensorflow as tf

logger = logging.getLogger(__name__)


class BaseMultiDriftOnline(BaseDetector, StateMixin):
    t: int = 0
    thresholds: np.ndarray
    backend: Literal['pytorch', 'tensorflow']
    online_state_keys: Tuple[str, ...]

    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            ert: float,
            window_size: int,
            preprocess_fn: Optional[Callable] = None,
            x_ref_preprocessed: bool = False,
            n_bootstraps: int = 1000,
            verbose: bool = True,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None,
    ) -> None:
        """
        Base class for multivariate online drift detectors.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        ert
            The expected run-time (ERT) in the absence of drift. For the multivariate detectors, the ERT is defined
            as the expected run-time from t=0.
        window_size
            The size of the sliding test-window used to compute the test-statistic.
            Smaller windows focus on responding quickly to severe drift, larger windows focus on
            ability to detect slight drift.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        x_ref_preprocessed
            Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only
            the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference
            data will also be preprocessed.
        n_bootstraps
            The number of bootstrap simulations used to configure the thresholds. The larger this is the
            more accurately the desired ERT will be targeted. Should ideally be at least an order of magnitude
            larger than the ert.
        verbose
            Whether or not to print progress during configuration.
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()

        if ert is None:
            logger.warning('No expected run-time set for the drift threshold. Need to set it to detect data drift.')

        self.ert = ert
        self.fpr = 1 / ert
        self.window_size = window_size

        # x_ref preprocessing
        self.x_ref_preprocessed = x_ref_preprocessed
        if preprocess_fn is not None and not isinstance(preprocess_fn, Callable):  # type: ignore[arg-type]
            raise ValueError("`preprocess_fn` is not a valid Callable.")
        if not self.x_ref_preprocessed and preprocess_fn is not None:
            self.x_ref = preprocess_fn(x_ref)
        else:
            self.x_ref = x_ref

        # Other attributes
        self.preprocess_fn = preprocess_fn
        self.n = len(x_ref)
        self.n_bootstraps = n_bootstraps  # nb of samples used to estimate thresholds
        self.verbose = verbose

        # store input shape for save and load functionality
        self.input_shape = get_input_shape(input_shape, x_ref)

        # set metadata
        self.meta['detector_type'] = 'drift'
        self.meta['data_type'] = data_type
        self.meta['online'] = True

    @abstractmethod
    def _configure_thresholds(self):
        pass

    @abstractmethod
    def _configure_ref_subset(self):
        pass

    @abstractmethod
    def _update_state(self, x_t: Union[np.ndarray, 'tf.Tensor', 'torch.Tensor']):
        pass

    def _preprocess_xt(self, x_t: Union[np.ndarray, Any]) -> np.ndarray:
        """
        Private method to preprocess a single test instance ready for _update_state.

        Parameters
        ----------
        x_t
            A single test instance to be preprocessed.

        Returns
        -------
        The preprocessed test instance `x_t`.
        """
        # preprocess if necessary
        if self.preprocess_fn is not None:
            x_t = x_t[None, :] if isinstance(x_t, np.ndarray) else [x_t]
            x_t = self.preprocess_fn(x_t)[0]
        return x_t[None, :]

    def get_threshold(self, t: int) -> float:
        """
        Return the threshold for timestep `t`.

        Parameters
        ----------
        t
            The timestep to return a threshold for.

        Returns
        -------
        The threshold at timestep `t`.
        """
        return self.thresholds[t] if t < self.window_size else self.thresholds[-1]

    def _initialise_state(self) -> None:
        """
        Initialise online state (the stateful attributes updated by `score` and `predict`).

        If a subclassed detector has additional online state, an additional `_initialise_state` should be defined,
        with a call to `super()._initialise_state()` included (see `LSDDDriftOnlineTorch._initialise_state()` for
        an example).
        """
        self.t = 0  # corresponds to a test set of ref data
        self.test_stats = np.array([])
        self.drift_preds = np.array([])

    def reset(self) -> None:
        """
        Deprecated reset method. This method will be repurposed or removed in the future. To reset the detector to
        its initial state (`t=0`) use :meth:`reset_state`.
        """
        self.reset_state()
        warnings.warn('This method is deprecated and will be removed/repurposed in the future. To reset the detector '
                      'to its initial state use `reset_state`.', DeprecationWarning)

    def reset_state(self) -> None:
        """
        Resets the detector to its initial state (`t=0`). This does not include reconfiguring thresholds.
        """
        self._initialise_state()

    def predict(self, x_t: Union[np.ndarray, Any], return_test_stat: bool = True,
                ) -> Dict[Dict[str, str], Dict[str, Union[int, float]]]:
        """
        Predict whether the most recent window of data has drifted from the reference data.

        Parameters
        ----------
        x_t
            A single instance to be added to the test-window.
        return_test_stat
            Whether to return the test statistic and threshold.

        Returns
        -------
        Dictionary containing ``'meta'`` and ``'data'`` dictionaries.
            - ``'meta'`` has the model's metadata.
            - ``'data'`` contains the drift prediction and optionally the test-statistic and threshold.
        """
        # Compute test stat and check for drift
        test_stat = self.score(x_t)
        threshold = self.get_threshold(self.t)
        drift_pred = int(test_stat > threshold)

        self.test_stats = np.concatenate([self.test_stats, np.array([test_stat])])
        self.drift_preds = np.concatenate([self.drift_preds, np.array([drift_pred])])

        # populate drift dict
        cd = concept_drift_dict()
        cd['meta'] = self.meta
        cd['data']['is_drift'] = drift_pred
        cd['data']['time'] = self.t
        cd['data']['ert'] = self.ert
        if return_test_stat:
            cd['data']['test_stat'] = test_stat
            cd['data']['threshold'] = threshold

        return cd


class BaseUniDriftOnline(BaseDetector, StateMixin):
    t: int = 0
    thresholds: np.ndarray
    online_state_keys: Tuple[str, ...]

    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            ert: float,
            window_sizes: List[int],
            preprocess_fn: Optional[Callable] = None,
            x_ref_preprocessed: bool = False,
            n_bootstraps: int = 1000,
            n_features: Optional[int] = None,
            verbose: bool = True,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None,
    ) -> None:
        """
        Base class for univariate online drift detectors. If n_features > 1, a multivariate correction is
        used to aggregate p-values during threshold configuration, thus allowing the requested expected run
        time (ERT) to be targeted. The multivariate correction assumes independence between the features.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        ert
            The expected run-time (ERT) in the absence of drift. For the univariate detectors, the ERT is defined
            as the expected run-time after the smallest window is full i.e. the run-time from t=min(windows_sizes)-1.
        window_sizes
            The sizes of the sliding test-windows used to compute the test-statistic.
            Smaller windows focus on responding quickly to severe drift, larger windows focus on
            ability to detect slight drift.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        x_ref_preprocessed
            Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only
            the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference
            data will also be preprocessed.
        n_bootstraps
            The number of bootstrap simulations used to configure the thresholds. The larger this is the
            more accurately the desired ERT will be targeted. Should ideally be at least an order of magnitude
            larger than the ert.
        n_features
            Number of features used in the statistical test. No need to pass it if no preprocessing takes place.
            In case of a preprocessing step, this can also be inferred automatically but could be more
            expensive to compute.
        verbose
            Whether or not to print progress during configuration.
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()

        if ert is None:
            logger.warning('No expected run-time set for the drift threshold. Need to set it to detect data drift.')

        self.ert = ert
        self.fpr = 1 / ert

        # Window sizes
        self.window_sizes = window_sizes
        self.max_ws = np.max(self.window_sizes)
        self.min_ws = np.min(self.window_sizes)

        # x_ref preprocessing
        self.x_ref_preprocessed = x_ref_preprocessed
        if preprocess_fn is not None and not isinstance(preprocess_fn, Callable):  # type: ignore[arg-type]
            raise ValueError("`preprocess_fn` is not a valid Callable.")
        if not self.x_ref_preprocessed and preprocess_fn is not None:
            self.x_ref = preprocess_fn(x_ref)
        else:
            self.x_ref = x_ref
        # Check the (optionally preprocessed) x_ref data is a 2D ndarray
        self.x_ref = self._check_x(self.x_ref, x_ref=True)

        # Other attributes
        self.preprocess_fn = preprocess_fn
        self.n = len(x_ref)
        self.n_bootstraps = n_bootstraps  # nb of samples used to estimate thresholds
        self.verbose = verbose

        # compute number of features for the univariate tests
        if isinstance(n_features, int):
            self.n_features = n_features
        elif not isinstance(preprocess_fn, Callable) or x_ref_preprocessed:
            # infer features from preprocessed reference data
            self.n_features = self.x_ref.reshape(self.x_ref.shape[0], -1).shape[-1]
        else:  # infer number of features after applying preprocessing step
            x = self.preprocess_fn(x_ref[0:1])
            self.n_features = x.reshape(x.shape[0], -1).shape[-1]

        # store input shape for save and load functionality
        self.input_shape = get_input_shape(input_shape, x_ref)

        # set metadata
        self.meta['detector_type'] = 'drift'
        self.meta['data_type'] = data_type
        self.meta['online'] = True

    @abstractmethod
    def _configure_thresholds(self):
        pass

    @abstractmethod
    def _configure_ref(self):
        pass

    @abstractmethod
    def _update_state(self, x_t: np.ndarray):
        pass

    def _check_x(self, x: Any, x_ref: bool = False) -> np.ndarray:
        """
        Check the type and shape of the data `x`, and coerces it to the correct shape if possible.

        Parameters
        ----------
        x
            The data to be checked.
        x_ref
            Whether `x` is a batch of reference data instances (if `True`), or a single test data instance (if `False`).

        Returns
        -------
        The checked data, coerced to be a np.ndarray of the correct shape.
        """
        # Check the type of x
        if isinstance(x, np.ndarray):
            pass
        elif isinstance(x, (int, float)):
            x = np.array([x])
        else:
            raise TypeError("Detectors expect data to be 2D np.ndarray's. If data is passed as another type, a "
                            "`preprocess_fn` should be given to convert this data to 2D np.ndarray's.")

        # Check the shape of x
        if x_ref:
            x = x.reshape(x.shape[0], -1)
        else:
            x = x.reshape(1, -1)
            if x.shape[1] != self.x_ref.shape[1]:
                raise ValueError("Dimensions do not match. `x` has shape (%d,%d) and `x_ref` has shape (%d,%d)."
                                 % (x.shape[0], x.shape[1], self.x_ref.shape[0], self.x_ref.shape[1]))
        return x

    def _preprocess_xt(self, x_t: Union[np.ndarray, Any]) -> np.ndarray:
        """
        Private method to preprocess a single test instance ready for _update_state.

        Parameters
        ----------
        x_t
            A single test instance to be preprocessed.

        Returns
        -------
        The preprocessed test instance `x_t`.
        """
        # preprocess if necessary
        if self.preprocess_fn is not None:
            x_t = x_t[None, :] if isinstance(x_t, np.ndarray) else [x_t]
            x_t = self.preprocess_fn(x_t)[0]
        # Now check the final data is a 2D ndarray
        x_t = self._check_x(x_t)
        return x_t

    def get_threshold(self, t: int) -> np.ndarray:
        """
        Return the threshold for timestep `t`.

        Parameters
        ----------
        t
            The timestep to return a threshold for.

        Returns
        -------
        The threshold at timestep `t`.
        """
        return self.thresholds[t] if t < len(self.thresholds) else self.thresholds[-1]

    def _initialise_state(self) -> None:
        """
        Initialise online state (the stateful attributes updated by `score` and `predict`).

        If a subclassed detector has additional online state, an additional `_initialise_state` should be defined,
        with a call to `super()._initialise_state()` included (see `CVMDriftOnlineTorch._initialise_state()` for
        an example).
        """
        self.t = 0
        self.xs = np.array([])
        self.test_stats = np.empty([0, len(self.window_sizes), self.n_features])
        self.drift_preds = np.array([])

    @abstractmethod
    def _check_drift(self, test_stats: np.ndarray, thresholds: np.ndarray) -> int:
        pass

    def reset(self) -> None:
        """
        Deprecated reset method. This method will be repurposed or removed in the future. To reset the detector to
        its initial state (`t=0`) use :meth:`reset_state`.
        """
        self.reset_state()
        warnings.warn('This method is deprecated and will be removed/repurposed in the future. To reset the detector '
                      'to its initial state use `reset_state`.', DeprecationWarning)

    def reset_state(self) -> None:
        """
        Resets the detector to its initial state (`t=0`). This does not include reconfiguring thresholds.
        """
        self._initialise_state()

    def predict(self, x_t: Union[np.ndarray, Any], return_test_stat: bool = True,
                ) -> Dict[Dict[str, str], Dict[str, Union[int, float]]]:
        """
        Predict whether the most recent window(s) of data have drifted from the reference data.

        Parameters
        ----------
        x_t
            A single instance to be added to the test-window(s).
        return_test_stat
            Whether to return the test statistic and threshold.

        Returns
        -------
        Dictionary containing ``'meta'`` and ``'data'`` dictionaries.
            - ``'meta'`` has the model's metadata.
            - ``'data'`` contains the drift prediction and optionally the test-statistic and threshold.
        """
        # Compute test stat and check for drift
        test_stats = self.score(x_t)
        thresholds = self.get_threshold(self.t - 1)  # Note t-1 here, has we wish to use the unconditional thresholds
        drift_pred = self._check_drift(test_stats, thresholds)

        # Update results attributes
        self.test_stats = np.concatenate([self.test_stats, test_stats[None, :, :]])
        self.drift_preds = np.concatenate([self.drift_preds, np.array([drift_pred])])

        # populate drift dict
        cd = concept_drift_dict()
        cd['meta'] = self.meta
        cd['data']['is_drift'] = drift_pred
        cd['data']['time'] = self.t
        cd['data']['ert'] = self.ert
        if return_test_stat:
            cd['data']['test_stat'] = test_stats
            cd['data']['threshold'] = thresholds

        return cd
