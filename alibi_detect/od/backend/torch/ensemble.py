from __future__ import annotations

import logging

from typing import Optional
from abc import ABC, abstractmethod

import torch
from torch.nn import Module
import numpy as np


logger = logging.getLogger(__name__)


class BaseTransformTorch(Module, ABC):
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


class FitMixinTorch:
    """Fit mixin

    Utility class that provides fitted checks for alibi-detect objects that require to be fit before use.

    TODO: this should be encorporated into alibi_detect/base.py FitMixinTorch once we can be sure that the
    behavour is compatible.
    """
    _fitted = False

    def __init__(self):
        super().__init__()

    def fit(self, X: torch.Tensor) -> BaseTransformTorch:
        self._fitted = True
        self._fit(X)
        return self

    def _fit(self, X: torch.Tensor):
        raise NotImplementedError()

    @torch.jit.ignore
    def check_fitted(self):
        if not self._fitted:
            # TODO: make our own NotFitted Error here!
            raise ValueError(f'{self.__class__.__name__} has not been fit!')


class BaseFittedTransformTorch(BaseTransformTorch, FitMixinTorch):
    """Base Fitted Transform class.

    Provides abstract methods for transforms that have an aditional
    fit step.
    """

    def __init__(self):
        BaseTransformTorch.__init__(self)
        FitMixinTorch.__init__(self)

    def transform(self, X: torch.Tensor):
        self.check_fitted()
        return self._transform(X)


class PValNormaliser(BaseFittedTransformTorch):
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


class ShiftAndScaleNormaliser(BaseFittedTransformTorch):
    """Maps scores to their normalised values.

    Needs to be fit on a reference dataset using fit. Subtracts the dataset mean and
    scales by the standard deviation.
    """
    def __init__(self):
        super().__init__()
        self.val_means = None
        self.val_scales = None

    def _fit(self, val_scores: torch.Tensor) -> BaseTransformTorch:
        self.val_means = val_scores.mean(0)[None, :]
        self.val_scales = val_scores.std(0)[None, :]

    def _transform(self, scores: torch.Tensor) -> torch.Tensor:
        return (scores - self.val_means)/self.val_scales


class TopKAggregator(BaseTransformTorch):
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


class AverageAggregator(BaseTransformTorch):
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


class MaxAggregator(BaseTransformTorch):
    """Takes the max score of a set of detectors in an ensemble."""
    def __init__(self):
        super().__init__()

    def _transform(self, scores: torch.Tensor) -> torch.Tensor:
        vals, _ = torch.max(scores, dim=-1)
        return vals


class MinAggregator(BaseTransformTorch):
    """Takes the min score of a set of detectors in an ensemble."""
    def __init__(self):
        super().__init__()

    def _transform(self, scores: torch.Tensor) -> torch.Tensor:
        vals, _ = torch.min(scores, dim=-1)
        return vals


class Accumulator(BaseFittedTransformTorch):
    def __init__(self,
                 normaliser: BaseFittedTransformTorch = None,
                 aggregator: BaseTransformTorch = AverageAggregator()):
        """Wraps a normaliser and aggregator into a single object.

        The accumulator wraps normalisers and aggregators into a single object.

        Parameters
        ----------
        normaliser
            normaliser that's an instance of BaseFittedTransformTorch. Maps the outputs of
            a set of detectors to a common range.
        aggregator
            aggregator extendng BaseTransformTorch. Maps outputs of the normaliser to
            single score.
        """
        super().__init__()
        self.normaliser = normaliser
        if self.normaliser is None:
            self.fitted = True
        self.aggregator = aggregator

    def _transform(self, X: torch.Tensor):
        if self.normaliser is not None:
            X = self.normaliser(X)
        X = self.aggregator(X)
        return X

    def _fit(self, X: torch.Tensor):
        if self.normaliser is not None:
            X = self.normaliser.fit(X)
