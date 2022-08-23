from __future__ import annotations

import logging
import numpy as np

from typing import Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseTransform(ABC):
    fitted = False

    def fit(self, X) -> BaseTransform:
        if not self.fitted and hasattr(self, '_fit'):
            self._fit(X)
            self.fitted = True
        return self

    def _fit(self, X):
        pass

    def transform(self, scores):
        if not self.fitted:
            raise Exception('Transform not fitted, call fit before calling transform!')
        self._transform(scores)

    @abstractmethod
    def _transform(self, scores):
        pass


class PValNormaliser(BaseTransform):
    def _fit(self, val_scores: np.ndarray):
        self.val_scores = val_scores

    def _transform(self, scores: np.ndarray) -> np.ndarray:
        p_vals = (
                1 + (scores[:,None,:] < self.val_scores[None,:,:]).sum(1)
            )/(len(self.val_scores)+1)
        return 1 - p_vals


class ShiftAndScaleNormaliser(BaseTransform):
    def _fit(self, val_scores: np.ndarray) -> BaseTransform:
        self.val_means = val_scores.mean(-1)[None,:]
        self.val_scales = val_scores.std(-1)[None,:]

    def _transform(self, scores: np.ndarray) -> np.ndarray:
        return (scores - self.val_means)/self.val_scales


class TopKAggregator(BaseTransform):
    def __init__(self, k: Optional[int]):
        self.k = k
        self.fitted = True

    def _transform(self, scores: np.ndarray) -> np.ndarray:
        if self.k is None:
            self.k = int(np.ceil(scores.shape[1]/2))
        return np.sort(scores, 1)[:, -self.k:].mean(-1)


class AverageAggregator(BaseTransform):
    def __init__(self, weights: Optional[np.ndarray] = None):
        self.weights = weights
        self.fitted = True

    def _transform(self, scores: np.ndarray) -> np.ndarray:
        if self.weights is None:
            m = scores.shape[-1]
            self.weights = np.ones(m)/m
        return scores @ self.weights


class MaxAggregator:
    def __init__(self):
        self.fitted = True

    def _transform(self, scores: np.ndarray) -> np.ndarray:
        return np.max(scores, axis=-1)


class MinAggregator:
    def __init__(self):
        self.fitted = True

    def _transform(self, scores: np.ndarray) -> np.ndarray:
        return np.min(scores, axis=-1)
