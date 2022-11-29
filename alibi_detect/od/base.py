from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any

from typing_extensions import Protocol, runtime_checkable
import numpy as np

from alibi_detect.base import BaseDetector


class OutlierDetector(BaseDetector, ABC):
    threshold_inferred = False
    ensemble = False

    def __init__(self):
        """ Base class for outlier detection algorithms."""
        super().__init__()
        self.meta['online'] = False
        self.meta['detector_type'] = 'outlier'

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        """
        Fit outlier detector to data.

        Parameters
        ----------
        X
            Reference data.
        """
        pass

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores of the instances in X.

        Parameters
        ----------
        X
            Data to score.

        Returns
        -------
        Anomaly scores. The higher the score, the more anomalous the instance.
        """
        pass

    @abstractmethod
    def infer_threshold(self, X: np.ndarray, fpr: float) -> None:
        """
        Infer the threshold for the outlier detector.

        Parameters
        ----------
        X
            Reference data.
        fpr
            False positive rate used to infer the threshold.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Predict whether the instances in X are outliers or not.

        Parameters
        ----------
        X
            Data to predict.

        Returns
        -------
        Dict with keys 'data' and 'meta'. 'data' contains the outlier scores. If threshold inference was performed,
        'data' also contains the outlier labels.
        """
        pass


# Using Protocols instead base classes for the backend classes. This is a bit more flexible and allows us to
# avoid the torch/tensorflow imports in the base class.
@runtime_checkable
class TransformProtocol(Protocol):
    """Protocol for transformer objects."""
    def transform(self, X):
        pass

    def _transform(self, X):
        pass


@runtime_checkable
class FittedTransformProtocol(TransformProtocol, Protocol):
    """Protocol for fitted transformer objects."""
    def fit(self, x_ref):
        pass

    def _fit(self, x_ref):
        pass

    def check_fitted(self):
        pass
