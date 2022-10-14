from __future__ import annotations

import logging
import numpy as np

from typing import Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseTransform(ABC):
    """Base Transform class.

    provides abstract methods for transform objects that map a numpy
    array.
    """
    def transform(self, X: np.ndarray):
        return self._transform(X)

    @abstractmethod
    def _transform(self, X: np.ndarray):
        """Applies class transform to numpy array

        Parameters
        ----------
        X
            numpy array to be transformed

        Raises
        ------
        NotImplementedError
            if _transform is not implimented on child class raise
            NotImplementedError
        """
        raise NotImplementedError()


class BaseFittedTransform(BaseTransform):
    """Base Fitted Transform class.

    Provides abstract methods for transforms that have an aditional
    fit step.
    """
    fitted = False

    def fit(self, X: np.ndarray) -> BaseTransform:
        if not self.fitted and hasattr(self, '_fit'):
            self._fit(X)
            self.fitted = True
        return self

    def _fit(self, X: np.ndarray):
        raise NotImplementedError()

    def transform(self, X: np.ndarray):
        if not self.fitted:
            raise ValueError('Transform not fitted, call fit before calling transform!')
        return self._transform(X)


class PValNormaliser(BaseTransform):
    """Maps scores from an ensemble of detectors to there p values.

    Needs to be fit on a reference dataset using fit. Transform counts the number of scores
    in the reference dataset that are greter than the score of interest and divides by the
    size of the reference dataset. Output is between 1 and 0. Small values are likely to be
    outliers.
    """
    def _fit(self, val_scores: np.ndarray):
        self.val_scores = val_scores

    def _transform(self, scores: np.ndarray) -> np.ndarray:
        p_vals = (
                1 + (scores[:, None, :] < self.val_scores[None, :, :]).sum(1)
            )/(len(self.val_scores)+1)
        return 1 - p_vals


class ShiftAndScaleNormaliser(BaseTransform):
    def _fit(self, val_scores: np.ndarray) -> BaseTransform:
        self.val_means = val_scores.mean(0)[None, :]
        self.val_scales = val_scores.std(0)[None, :]

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


class MaxAggregator(BaseTransform):
    def __init__(self):
        self.fitted = True

    def _transform(self, scores: np.ndarray) -> np.ndarray:
        return np.max(scores, axis=-1)


class MinAggregator(BaseTransform):
    def __init__(self):
        self.fitted = True

    def _transform(self, scores: np.ndarray) -> np.ndarray:
        return np.min(scores, axis=-1)
