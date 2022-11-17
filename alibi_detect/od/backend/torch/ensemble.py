from __future__ import annotations

import logging

from typing import Optional
from abc import ABC, abstractmethod

import torch
from torch.nn import Module
import numpy as np


logger = logging.getLogger(__name__)


class BaseTransform(Module, ABC):
    """Base Transform class.

    provides abstract methods for transform objects that map a numpy
    array.
    """
    def __init__(self):
        super().__init__()

    def transform(self, X: torch.Tensor):
        return self._transform(X)

    @abstractmethod
    def _transform(self, X: torch.Tensor):
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

    def forward(self, X: torch.Tensor):
        return self.transform(X=X)


class BaseFittedTransform(BaseTransform):
    """Base Fitted Transform class.

    Provides abstract methods for transforms that have an aditional
    fit step.
    """
    fitted = False

    def __init__(self):
        super().__init__()

    def fit(self, X: torch.Tensor) -> BaseTransform:
        if not self.fitted and hasattr(self, '_fit'):
            self._fit(X)
            self.fitted = True
        return self

    def _fit(self, X: torch.Tensor):
        raise NotImplementedError()

    def transform(self, X: torch.Tensor):
        if not self.fitted:
            raise ValueError('Transform not fitted, call fit before calling transform!')
        return self._transform(X)


class PValNormaliser(BaseFittedTransform):
    """Maps scores to there p values.

    Needs to be fit on a reference dataset using fit. Transform counts the number of scores
    in the reference dataset that are greter than the score of interest and divides by the
    size of the reference dataset. Output is between 1 and 0. Small values are likely to be
    outliers.
    """
    def __init__(self):
        super().__init__()
        self.val_scores = None

    def _fit(self, val_scores: torch.Tensor):
        self.val_scores = val_scores

    def _transform(self, scores: torch.Tensor) -> torch.Tensor:
        p_vals = (
                1 + (scores[:, None, :] < self.val_scores[None, :, :]).sum(1)
            )/(len(self.val_scores)+1)
        return 1 - p_vals


class ShiftAndScaleNormaliser(BaseFittedTransform):
    """Maps scores to their normalised values.

    Needs to be fit on a reference dataset using fit. Subtracts the dataset mean and
    scales by the standard deviation.
    """
    def __init__(self):
        super().__init__()
        self.val_means = None
        self.val_scales = None

    def _fit(self, val_scores: torch.Tensor) -> BaseTransform:
        self.val_means = val_scores.mean(0)[None, :]
        self.val_scales = val_scores.std(0)[None, :]

    def _transform(self, scores: torch.Tensor) -> torch.Tensor:
        return (scores - self.val_means)/self.val_scales


class TopKAggregator(BaseTransform):
    def __init__(self, k: Optional[int] = None):
        """Takes the mean of the top k scores.

        Parameters
        ----------
        k
            number of scores to take the mean of. If `k` is left `None` then will be set to
            half the number of scores passed in the forward call.
        """
        super().__init__()
        self.k = k

    def _transform(self, scores: torch.Tensor) -> torch.Tensor:
        if self.k is None:
            self.k = int(np.ceil(scores.shape[1]/2))
        sorted_scores, _ = torch.sort(scores, 1)
        return sorted_scores[:, -self.k:].mean(-1)


class AverageAggregator(BaseTransform):
    """Averages the scores of the detectors in an ensemble.

    Parameters
    ----------
    weights
        Optional parameter to weight the scores.
    """
    def __init__(self, weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.weights = weights

    def _transform(self, scores: torch.Tensor) -> torch.Tensor:
        if self.weights is None:
            m = scores.shape[-1]
            self.weights = torch.ones(m)/m
        return scores @ self.weights


class MaxAggregator(BaseTransform):
    """Takes the max score of a set of detectors in an ensemble."""
    def __init__(self):
        super().__init__()

    def _transform(self, scores: torch.Tensor) -> torch.Tensor:
        vals, _ = torch.max(scores, dim=-1)
        return vals


class MinAggregator(BaseTransform):
    """Takes the min score of a set of detectors in an ensemble."""
    def __init__(self):
        super().__init__()

    def _transform(self, scores: torch.Tensor) -> torch.Tensor:
        vals, _ = torch.min(scores, dim=-1)
        return vals


class Accumulator(BaseFittedTransform):
    def __init__(self, normaliser: BaseFittedTransform, aggregator: BaseTransform):
        super().__init__()
        self.normaliser = normaliser
        self.aggregator = aggregator

    def _transform(self, X: torch.Tensor):
        X = self.normaliser(X)
        return self.aggregator(X)

    def _fit(self, X: torch.Tensor):
        return self.normaliser.fit(X)