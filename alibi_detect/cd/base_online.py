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


class BaseMMDDriftOnline(BaseDetector):
    def __init__(
            self,
            x_ref: np.ndarray,
            ert: float,
            window_size: int,
            preprocess_x_ref: bool = True,
            preprocess_fn: Optional[Callable] = None,
            sigma: Optional[np.ndarray] = None,
            n_bootstraps: int = 1000,
            input_shape: Optional[tuple] = None,
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
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()

        if ert is None:
            logger.warning('No expected run-time set for the drift threshold. Need to set it to detect data drift.')

        self.ert = ert
        self.fpr = 1/ert
        self.window_size = window_size

        if preprocess_x_ref and isinstance(preprocess_fn, Callable):  # type: ignore
            self.x_ref = preprocess_fn(x_ref)
        else:
            self.x_ref = x_ref
        self.preprocess_x_ref = preprocess_x_ref
        self.preprocess_fn = preprocess_fn
        self.n = x_ref.shape[0]  # type: ignore
        self.n_bootstraps = n_bootstraps  # nb of samples used to estimate thresholds
        self.sigma = sigma

        # store input shape for save and load functionality
        self.input_shape = input_shape if isinstance(input_shape, tuple) else x_ref.shape[1:]

        # set metadata
        self.meta['detector_type'] = 'online'
        self.meta['data_type'] = data_type

    @abstractmethod
    def kernel_matrix(self, x: Union['torch.Tensor', 'tf.Tensor'], y: Union['torch.Tensor', 'tf.Tensor']) \
            -> Union['torch.Tensor', 'tf.Tensor']:
        pass

    @abstractmethod
    def _configure_thresholds(self):
        pass

    @abstractmethod
    def _configure_ref_subset(self):
        pass

    def get_threshold(self, t: int) -> Union[float, None]:
        if t < self.window_size:
            threshold = None
        if self.window_size <= t < 2*self.window_size:
            threshold = self.thresholds[t-self.window_size]
        if self.window_size >= 2*self.window_size:
            threshold = self.thresholds[-1]
        return threshold

    def _initialise(self) -> None:
        self.t = -1  # 0 will correspond to first observation
        self.test_stats = np.array([])
        self.thresholds = np.array([])
        self._configure_ref_subset()

    def reset(self) -> None:
        self._initialise()

    # @abstractmethod
    # def _update(self, x_t: np.ndarray) -> float:
    #     pass

    def predict(self, x_t: np.ndarray,  return_test_stat: bool = True,
                ) -> Dict[Dict[str, str], Dict[str, Union[int, float]]]:
        """
        Predict whether a of data has drifted from the reference data.

        Parameters
        ----------
        x_t
            A single instance.
        return_test_stat
            Whether to return the test-statistic S_t.
        return_threshold
            Whether to return the threshold b_t.

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the drift prediction and optionally the test-statistic and threshold.
        """
        self.t += 1

        # preprocess if necessary
        if self.preprocess_x_ref and isinstance(self.preprocess_fn, Callable):
            x_t = self.preprocess_fn(x_t)

        # update test window and return updated test stat
        test_stat = self.score(x_t)
        threshold = self.get_threshold(self.t)

        drift_pred = int(self.test_stat < threshold)

        # populate drift dict
        # TODO: add instance level feedback
        cd = concept_drift_dict()
        cd['meta'] = self.meta
        cd['data']['is_drift'] = drift_pred
        cd['data']['time'] = self.t
        cd['data']['ert'] = self.ert
        if return_test_stat:
            cd['data']['test_stat'] = test_stat
            cd['data']['threshold'] = threshold

        return cd
