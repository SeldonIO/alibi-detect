from abc import abstractmethod
import logging
import numpy as np
from typing import Callable, Dict,  Optional, Union
from alibi_detect.base import BaseDetector, concept_drift_dict
from alibi_detect.utils.frameworks import has_pytorch, has_tensorflow

if has_pytorch:
    import torch  # noqa F401

if has_tensorflow:
    import tensorflow as tf  # noqa F401

logger = logging.getLogger(__name__)


class BaseDriftOnline(BaseDetector):
    def __init__(
            self,
            x_ref: np.ndarray,
            ert: float,
            window_size: int,
            preprocess_fn: Optional[Callable] = None,
            n_bootstraps: int = 1000,
            verbose: bool = True,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None,
    ) -> None:
        """
        Base class for online drift detectors.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        ert
            The expected run-time (ERT) in the absence of drift.
        window_size
            The size of the sliding test-window used to compute the test-statistic.
            Smaller windows focus on responding quickly to severe drift, larger windows focus on
            ability to detect slight drift.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
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
        self.fpr = 1/ert
        self.window_size = window_size

        if isinstance(preprocess_fn, Callable):  # type: ignore
            self.x_ref = preprocess_fn(x_ref)
        else:
            self.x_ref = x_ref
        self.preprocess_fn = preprocess_fn
        self.n = x_ref.shape[0]  # type: ignore
        self.n_bootstraps = n_bootstraps  # nb of samples used to estimate thresholds
        self.verbose = verbose

        # store input shape for save and load functionality
        self.input_shape = input_shape if isinstance(input_shape, tuple) else x_ref.shape[1:]

        # set metadata
        self.meta['detector_type'] = 'online'
        self.meta['data_type'] = data_type

    @abstractmethod
    def _configure_thresholds(self):
        pass

    @abstractmethod
    def _configure_ref_subset(self):
        pass

    def get_threshold(self, t: int) -> Union[float, None]:
        return self.thresholds[t] if t < self.window_size else self.thresholds[-1]  # type: ignore

    def _initialise(self) -> None:
        self.t = 0  # corresponds to a test set of ref data
        self.test_stats = np.array([])
        self.drift_preds = np.array([])
        self._configure_ref_subset()

    def reset(self) -> None:
        "Resets the detector but does not reconfigure thresholds."
        self._initialise()

    def predict(self, x_t: np.ndarray,  return_test_stat: bool = True,
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
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the drift prediction and optionally the test-statistic and threshold.
        """
        self.t += 1

        # preprocess if necessary
        if isinstance(self.preprocess_fn, Callable):  # type: ignore
            x_t = self.preprocess_fn(x_t[None, :])[0]

        # update test window and return updated test stat
        test_stat = self.score(x_t)
        threshold = self.get_threshold(self.t)
        drift_pred = 0 if test_stat is None else int(test_stat > threshold)

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
