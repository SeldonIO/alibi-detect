from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import logging
from typing import Dict
from typing_extensions import Protocol, runtime_checkable
from alibi_detect.base import BaseDetector

logger = logging.getLogger(__name__)


class OutlierDetector(BaseDetector, ABC):
    """ Base class for outlier detection algorithms. """
    threshold_inferred = False
    ensemble = False

    def __init__(self):
        super().__init__()
        self.meta['online'] = False
        self.meta['detector_type'] = 'outlier'

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        pass

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def infer_threshold(self, X: np.ndarray, fpr: float) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        pass


@runtime_checkable
class TransformProtocol(Protocol):
    def transform(self, X):
        pass

    def _transform(self, X):
        pass


@runtime_checkable
class FittedTransformProtocol(TransformProtocol, Protocol):
    def fit(self, x_ref):
        pass

    def _fit(self, x_ref):
        pass

    def check_fitted(self):
        pass
