from __future__ import annotations

import logging

from typing import Optional
from abc import ABC, abstractmethod

import torch
from torch.nn import Module
import numpy as np


logger = logging.getLogger(__name__)


class BaseTransformTorch(Module, ABC):
    def __init__(self):
        """Base Transform class.

        provides abstract methods for transform objects that map a numpy
        array.
        """
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
    _fitted = False

    def __init__(self):
        """Fit mixin

        Utility class that provides fitted checks for alibi-detect objects that require to be fit before use.
        """
        super().__init__()

    def fit(self, X: torch.Tensor) -> FitMixinTorch:
        self._fitted = True
        self._fit(X)
        return self

    def _fit(self, X: torch.Tensor):
        """Fit on `X` tensor.

        This method should be overidden on child classes.

        Parameters
        ----------
        X
            Reference `torch.Tensor` for fitting object.

        Raises
        ------
        NotImplementedError
            Raised if unimplimented.
        """
        raise NotImplementedError()

    @torch.jit.ignore
    def check_fitted(self):
        """Raises error if parent object instance has not been fit.

        Raises
        ------
        ValueError
            Raised if method called and object has not been fit.
        """
        if not self._fitted:
            # TODO: make our own NotFitted Error here!
            raise ValueError(f'{self.__class__.__name__} has not been fit!')


class BaseFittedTransformTorch(BaseTransformTorch, FitMixinTorch):
    def __init__(self):
        """Base Fitted Transform class.

        Extends BaseTransfrom with fit functionality. Ensures that transform has been fit prior to
        applying transform.
        """
        BaseTransformTorch.__init__(self)
        FitMixinTorch.__init__(self)

    def transform(self, X: torch.Tensor):
        """Checks to make sure transform has been fitted and then applies trasform to input tensor.

        Parameters
        ----------
        X
            `torch.Tensor` being transformed.

        Returns
        -------
        transformed `torch.Tensor`.
        """
        self.check_fitted()
        return self._transform(X)


class PValNormaliser(BaseFittedTransformTorch):
    def __init__(self):
        """Maps scores to there p values.

        Needs to be fit (see py:obj:alibi_detect.od.backend.torch.ensemble.BaseFittedTransformTorch).
        Transform counts the number of scores in the reference dataset that are greter than the score
        of interest and divides by the size of the reference dataset. Output is between 1 and 0. Small
        values are likely to be outliers.
        """
        super().__init__()
        self.val_scores = None

    def _fit(self, val_scores: torch.Tensor) -> PValNormaliser:
        """Fit transform on scores.

        Parameters
        ----------
        val_scores
            score outputs of ensemble of detectors applied to reference data.

        Returns
        -------
        self
        """
        self.val_scores = val_scores
        return self

    def _transform(self, scores: torch.Tensor) -> torch.Tensor:
        """Transform scores to 1 - p values.

        Parameters
        ----------
        scores
            `Torch.Tensor` of scores from ensemble of detectors.

        Returns
        -------
        `Torch.Tensor` of 1 - p values.
        """
        p_vals = (
                1 + (scores[:, None, :] < self.val_scores[None, :, :]).sum(1)
            )/(len(self.val_scores)+1)
        return 1 - p_vals


class ShiftAndScaleNormaliser(BaseFittedTransformTorch):
    def __init__(self):
        """Maps scores to their normalised values.

        Needs to be fit (see py:obj:alibi_detect.od.backend.torch.ensemble.BaseFittedTransformTorch).
        Subtracts the dataset mean and scales by the standard deviation.
        """
        super().__init__()
        self.val_means = None
        self.val_scales = None

    def _fit(self, val_scores: torch.Tensor) -> ShiftAndScaleNormaliser:
        """Computes the mean and standard deviation of the scores and stores them.

        Parameters
        ----------
        val_scores
            `Torch.Tensor` of scores from ensemble of detectors.

        Returns
        -------
        self
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
        """Takes the mean of the top k scores.

        Parameters
        ----------
        scores
            `Torch.Tensor` of scores from ensemble of detectors.

        Returns
        -------
        `Torch.Tensor` of mean of top k scores.
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
            Optional parameter to weight the scores. If `weights` is left `None` then will be set to
            a vector of ones.
        """
        super().__init__()
        self.weights = weights

    def _transform(self, scores: torch.Tensor) -> torch.Tensor:
        """Averages the scores of the detectors in an ensemble. If weights where passed in the init
        then these are used to weight the scores.

        Parameters
        ----------
        scores
            `Torch.Tensor` of scores from ensemble of detectors.

        Returns
        -------
        `Torch.Tensor` of mean of scores.
        """
        if self.weights is None:
            m = scores.shape[-1]
            self.weights = torch.ones(m)/m
        return scores @ self.weights


class MaxAggregator(BaseTransformTorch):
    def __init__(self):
        """Takes the maximum of the scores of the detectors in an ensemble."""
        super().__init__()

    def _transform(self, scores: torch.Tensor) -> torch.Tensor:
        """Takes the max score of a set of detectors in an ensemble.

        Parameters
        ----------
        scores
            `Torch.Tensor` of scores from ensemble of detectors.

        Returns
        -------
        `Torch.Tensor` of max of scores.
        """
        vals, _ = torch.max(scores, dim=-1)
        return vals


class MinAggregator(BaseTransformTorch):
    def __init__(self):
        """Takes the min score of a set of detectors in an ensemble."""
        super().__init__()

    def _transform(self, scores: torch.Tensor) -> torch.Tensor:
        """Takes the min score of a set of detectors in an ensemble.

        Parameters
        ----------
        scores
            `Torch.Tensor` of scores from ensemble of detectors.

        Returns
        -------
        `Torch.Tensor` of min of scores.
        """
        vals, _ = torch.min(scores, dim=-1)
        return vals


class Accumulator(BaseFittedTransformTorch):
    def __init__(self,
                 normaliser: Optional[BaseFittedTransformTorch] = None,
                 aggregator: BaseTransformTorch = AverageAggregator()):
        """Accumulates the scores of the detectors in an ensemble. Can be used to normalise and aggregate
        the scores from an ensemble of detectors.

        Parameters
        ----------
        normaliser
            `BaseFittedTransformTorch` object to normalise the scores. If `None` then no normalisation
            is applied.
        aggregator
            `BaseTransformTorch` object to aggregate the scores.
        """
        super().__init__()
        self.normaliser = normaliser
        if self.normaliser is None:
            self.fitted = True
        self.aggregator = aggregator

    def _transform(self, X: torch.Tensor):
        """Apply the normaliser and aggregator to the scores.

        Parameters
        ----------
        X
            `Torch.Tensor` of scores from ensemble of detectors.

        Returns
        -------
        `Torch.Tensor` of aggregated and normalised scores.
        """
        if self.normaliser is not None:
            X = self.normaliser(X)
        X = self.aggregator(X)
        return X

    def _fit(self, X: torch.Tensor):
        """Fit the normaliser to the scores.

        Parameters
        ----------
        X
            `Torch.Tensor` of scores from ensemble of detectors.
        """
        if self.normaliser is not None:
            self.normaliser.fit(X)
