from abc import ABC, abstractmethod
from typing import Optional
from typing_extensions import Self

import torch
import numpy as np
from torch.nn import Module

from alibi_detect.exceptions import NotFittedError


class BaseTransformTorch(Module):
    def __init__(self):
        """Base Transform class.

        provides abstract methods for transform objects that map `torch` tensors.
        """
        super().__init__()

    def transform(self, x: torch.Tensor):
        """Public transform method.

        Parameters
        ----------
        x
            `torch.Tensor` array to be transformed
        """
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)


class FitMixinTorch(ABC):
    fitted = False

    @abstractmethod
    def fit(self, x_ref: torch.Tensor) -> Self:
        """Abstract fit method.

        Parameters
        ----------
        x
            `torch.Tensor` to fit object on.
        """
        pass

    def _set_fitted(self) -> Self:
        """Sets the fitted attribute to True.

        Should be called within each transform method.
        """
        self.fitted = True
        return self

    def check_fitted(self):
        """Checks to make sure object has been fitted.

        Raises
        ------
        NotFittedError
            Raised if method called and object has not been fit.
        """
        if not torch.jit.is_scripting():
            self._check_fitted()

    @torch.jit.unused
    def _check_fitted(self):
        """Raises error if parent object instance has not been fit."""
        if not self.fitted:
            raise NotFittedError(self.__class__.__name__)


class PValNormalizer(BaseTransformTorch, FitMixinTorch):
    def __init__(self):
        """Maps scores to there p-values.

        Needs to be fit (see :py:obj:`~alibi_detect.od.pytorch.ensemble.BaseFittedTransformTorch`).
        Returns the proportion of scores in the reference dataset that are greater than the score of
        interest. Output is between ``1`` and ``0``. Small values are likely to be outliers.
        """
        super().__init__()
        self.val_scores = None

    def fit(self, val_scores: torch.Tensor) -> Self:
        """Fit transform on scores.

        Parameters
        ----------
        val_scores
            score outputs of ensemble of detectors applied to reference data.
        """
        self.val_scores = val_scores
        return self._set_fitted()

    def transform(self, scores: torch.Tensor) -> torch.Tensor:
        """Transform scores to 1 - p-values.

        Parameters
        ----------
        scores
            `Torch.Tensor` of scores from ensemble of detectors.

        Returns
        -------
        `Torch.Tensor` of 1 - p-values.
        """
        self.check_fitted()
        less_than_val_scores = scores[:, None, :] < self.val_scores[None, :, :]
        p_vals = (1 + less_than_val_scores.sum(1))/(len(self.val_scores) + 1)
        return 1 - p_vals


class ShiftAndScaleNormalizer(BaseTransformTorch, FitMixinTorch):
    def __init__(self):
        """Maps scores to their normalized values.

        Needs to be fit (see :py:obj:`~alibi_detect.od.pytorch.ensemble.BaseFittedTransformTorch`).
        Subtracts the dataset mean and scales by the standard deviation.
        """
        super().__init__()
        self.val_means = None
        self.val_scales = None

    def fit(self, val_scores: torch.Tensor) -> Self:
        """Computes the mean and standard deviation of the scores and stores them.

        Parameters
        ----------
        val_scores
            `Torch.Tensor` of scores from ensemble of detectors.
        """
        self.val_means = val_scores.mean(0)[None, :]
        self.val_scales = val_scores.std(0)[None, :]
        return self._set_fitted()

    def transform(self, scores: torch.Tensor) -> torch.Tensor:
        """Transform scores to normalized values. Subtracts the mean and scales by the standard deviation.

        Parameters
        ----------
        scores
            `Torch.Tensor` of scores from ensemble of detectors.

        Returns
        -------
        `Torch.Tensor` of normalized scores.
        """
        self.check_fitted()
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

    def transform(self, scores: torch.Tensor) -> torch.Tensor:
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
        if weights is not None and not np.isclose(weights.sum(), 1):
            raise ValueError("Weights must sum to 1.")
        self.weights = weights

    def transform(self, scores: torch.Tensor) -> torch.Tensor:
        """Averages the scores of the detectors in an ensemble. If weights were passed in the `__init__`
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
            self.weights = torch.ones(m, device=scores.device)/m
        return scores @ self.weights


class MaxAggregator(BaseTransformTorch):
    def __init__(self):
        """Takes the maximum of the scores of the detectors in an ensemble."""
        super().__init__()

    def transform(self, scores: torch.Tensor) -> torch.Tensor:
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

    def transform(self, scores: torch.Tensor) -> torch.Tensor:
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


class Ensembler(BaseTransformTorch, FitMixinTorch):
    def __init__(self,
                 normalizer: Optional[BaseTransformTorch] = None,
                 aggregator: BaseTransformTorch = None):
        """An Ensembler applies normalization and aggregation operations to the scores of an ensemble of detectors.

        Parameters
        ----------
        normalizer
            `BaseFittedTransformTorch` object to normalize the scores. If ``None`` then no normalization
            is applied.
        aggregator
            `BaseTransformTorch` object to aggregate the scores. If ``None`` defaults to `AverageAggregator`.
        """
        super().__init__()
        self.normalizer = normalizer
        if self.normalizer is None:
            self.fitted = True
        if aggregator is None:
            aggregator = AverageAggregator()
        self.aggregator = aggregator

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the normalizer and aggregator to the scores.

        Parameters
        ----------
        x
            `Torch.Tensor` of scores from ensemble of detectors.

        Returns
        -------
        `Torch.Tensor` of aggregated and normalized scores.
        """
        if self.normalizer is not None:
            x = self.normalizer(x)
        x = self.aggregator(x)
        return x

    def fit(self, x: torch.Tensor) -> Self:
        """Fit the normalizer to the scores.

        Parameters
        ----------
        x
            `Torch.Tensor` of scores from ensemble of detectors.
        """
        if self.normalizer is not None:
            self.normalizer.fit(x)  # type: ignore
        return self._set_fitted()
