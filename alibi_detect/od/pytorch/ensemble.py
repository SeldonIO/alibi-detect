from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

import torch
import numpy as np
from torch.nn import Module

from alibi_detect.od.base import NotFitException


class BaseTransformTorch(Module, ABC):
    def __init__(self):
        """Base Transform class.

        provides abstract methods for transform objects that map `torch` tensors.
        """
        super().__init__()

    def transform(self, x: torch.Tensor):
        return self._transform(x)

    @abstractmethod
    def _transform(self, x: torch.Tensor):
        """Applies class transform to `torch.Tensor`

        Parameters
        ----------
        x
            `torch.Tensor` array to be transformed
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x=x)


class FitMixinTorch(ABC):
    _fitted = False

    def __init__(self):
        """Fit mixin

        Utility class that provides fitted checks for alibi-detect objects that require to be fit before use.
        """
        super().__init__()

    def fit(self, x: torch.Tensor) -> FitMixinTorch:
        self._fitted = True
        self._fit(x)
        return self

    @abstractmethod
    def _fit(self, x: torch.Tensor):
        """Fit on `x` tensor.

        This method should be overidden on child classes.

        Parameters
        ----------
        x
            Reference `torch.Tensor` for fitting object.
        """
        pass

    @torch.jit.unused
    def check_fitted(self):
        """Raises error if parent object instance has not been fit.

        Raises
        ------
        NotFitException
            Raised if method called and object has not been fit.
        """
        if not self._fitted:
            raise NotFitException(f'{self.__class__.__name__} has not been fit!')


class BaseFittedTransformTorch(BaseTransformTorch, FitMixinTorch):
    def __init__(self):
        """Base Fitted Transform class.

        Extends `BaseTransfrom` with fit functionality. Ensures that transform has been fit prior to
        applying transform.
        """
        BaseTransformTorch.__init__(self)
        FitMixinTorch.__init__(self)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Checks to make sure transform has been fitted and then applies trasform to input tensor.

        Parameters
        ----------
        x
            `torch.Tensor` being transformed.

        Returns
        -------
        transformed `torch.Tensor`.
        """
        if not torch.jit.is_scripting():
            self.check_fitted()
        return self._transform(x)


class PValNormalizer(BaseFittedTransformTorch):
    def __init__(self):
        """Maps scores to there p-values.

        Needs to be fit (see :py:obj:`~alibi_detect.od.pytorch.ensemble.BaseFittedTransformTorch`).
        Returns the proportion of scores in the reference dataset that are greater than the score of
        interest. Output is between ``1`` and ``0``. Small values are likely to be outliers.
        """
        super().__init__()
        self.val_scores = None

    def _fit(self, val_scores: torch.Tensor) -> PValNormalizer:
        """Fit transform on scores.

        Parameters
        ----------
        val_scores
            score outputs of ensemble of detectors applied to reference data.

        Returns
        -------
        `self`
        """
        self.val_scores = val_scores
        return self

    def _transform(self, scores: torch.Tensor) -> torch.Tensor:
        """Transform scores to 1 - p-values.

        Parameters
        ----------
        scores
            `Torch.Tensor` of scores from ensemble of detectors.

        Returns
        -------
        `Torch.Tensor` of 1 - p-values.
        """
        p_vals = (
                1 + (scores[:, None, :] < self.val_scores[None, :, :]).sum(1)
            )/(len(self.val_scores)+1)
        return 1 - p_vals


class ShiftAndScaleNormalizer(BaseFittedTransformTorch):
    def __init__(self):
        """Maps scores to their normalised values.

        Needs to be fit (see :py:obj:`~alibi_detect.od.pytorch.ensemble.BaseFittedTransformTorch`).
        Subtracts the dataset mean and scales by the standard deviation.
        """
        super().__init__()
        self.val_means = None
        self.val_scales = None

    def _fit(self, val_scores: torch.Tensor) -> ShiftAndScaleNormalizer:
        """Computes the mean and standard deviation of the scores and stores them.

        Parameters
        ----------
        val_scores
            `Torch.Tensor` of scores from ensemble of detectors.

        Returns
        -------
        `self`
        """
        self.val_means = val_scores.mean(0)[None, :]
        self.val_scales = val_scores.std(0)[None, :]
        return self

    def _transform(self, scores: torch.Tensor) -> torch.Tensor:
        """Transform scores to normalised values. Subtracts the mean and scales by the standard deviation.

        Parameters
        ----------
        scores
            `Torch.Tensor` of scores from ensemble of detectors.

        Returns
        -------
        `Torch.Tensor` of normalised scores.
        """
        return (scores - self.val_means)/self.val_scales


class TopKAggregator(BaseTransformTorch):
    def __init__(self, k: Optional[int] = None):
        """Takes the mean of the top `k` scores.

        Parameters
        ----------
        k
            number of scores to take the mean of. If `k` is left ``None`` then will be set to
            half the number of scores passed in the forward call.
        """
        super().__init__()
        self.k = k

    def _transform(self, scores: torch.Tensor) -> torch.Tensor:
        """Takes the mean of the top `k` scores.

        Parameters
        ----------
        scores
            `Torch.Tensor` of scores from ensemble of detectors.

        Returns
        -------
        `Torch.Tensor` of mean of top `k` scores.
        """
        if self.k is None:
            self.k = int(np.ceil(scores.shape[1]/2))
        sorted_scores, _ = torch.sort(scores, 1)
        return sorted_scores[:, -self.k:].mean(-1)


class AverageAggregator(BaseTransformTorch):
    def __init__(self, weights: Optional[torch.Tensor] = None):
        """Averages the scores of the detectors in an ensemble.

        Parameters
        ----------
        weights
            Optional parameter to weight the scores. If `weights` is left ``None`` then will be set to
            a vector of ones.

        Raises
        ------
        ValueError
            If `weights` does not sum to ``1``.
        """
        super().__init__()
        if weights is not None and weights.sum() != 1:
            raise ValueError("Weights must sum to 1.")
        self.weights = weights

    def _transform(self, scores: torch.Tensor) -> torch.Tensor:
        """Averages the scores of the detectors in an ensemble. If weights where passed in the `__init__`
        then these are used to weight the scores.
        ----------
        scores
            `Torch.Tensor` of scores from ensemble of detectors.

        Returns
        -------
        `Torch.Tensor` of mean of scores.
        """
        if self.weights is None:
            m = scores.shape[-1]
            self.weights = torch.ones(m, device=scores.device)/m
        return scores @ self.weights


class MaxAggregator(BaseTransformTorch):
    def __init__(self):
        """Takes the maximum of the scores of the detectors in an ensemble."""
        super().__init__()

    def _transform(self, scores: torch.Tensor) -> torch.Tensor:
        """Takes the maximum score of a set of detectors in an ensemble.

        Parameters
        ----------
        scores
            `Torch.Tensor` of scores from ensemble of detectors.

        Returns
        -------
        `Torch.Tensor` of maximum scores.
        """
        vals, _ = torch.max(scores, dim=-1)
        return vals


class MinAggregator(BaseTransformTorch):
    def __init__(self):
        """Takes the minimum score of a set of detectors in an ensemble."""
        super().__init__()

    def _transform(self, scores: torch.Tensor) -> torch.Tensor:
        """Takes the minimum score of a set of detectors in an ensemble.

        Parameters
        ----------
        scores
            `Torch.Tensor` of scores from ensemble of detectors.

        Returns
        -------
        `Torch.Tensor` of minimum scores.
        """
        vals, _ = torch.min(scores, dim=-1)
        return vals


class Accumulator(BaseFittedTransformTorch):
    def __init__(self,
                 normalizer: Optional[BaseFittedTransformTorch] = None,
                 aggregator: BaseTransformTorch = AverageAggregator()):
        """Accumulates the scores of the detectors in an ensemble. Can be used to normalise and aggregate
        the scores from an ensemble of detectors.

        Parameters
        ----------
        normalizer
            `BaseFittedTransformTorch` object to normalise the scores. If ``None`` then no normalisation
            is applied.
        aggregator
            `BaseTransformTorch` object to aggregate the scores.
        """
        super().__init__()
        self.normalizer = normalizer
        if self.normalizer is None:
            self.fitted = True
        self.aggregator = aggregator

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the normalizer and aggregator to the scores.

        Parameters
        ----------
        x
            `Torch.Tensor` of scores from ensemble of detectors.

        Returns
        -------
        `Torch.Tensor` of aggregated and normalised scores.
        """
        if self.normalizer is not None:
            x = self.normalizer(x)
        x = self.aggregator(x)
        return x

    def _fit(self, x: torch.Tensor):
        """Fit the normalizer to the scores.

        Parameters
        ----------
        x
            `Torch.Tensor` of scores from ensemble of detectors.
        """
        if self.normalizer is not None:
            self.normalizer.fit(x)