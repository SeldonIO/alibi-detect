from __future__ import annotations
from abc import ABC, abstractmethod
import torch

import logging

logger = logging.getLogger(__name__)


class TorchOutlierDetector(torch.nn.Module, ABC):
    """ Base class for torch backend outlier detection algorithms."""
    threshold_inferred = False

    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, X: torch.Tensor) -> None:
        pass

    @abstractmethod
    def score(self, X: torch.Tensor) -> torch.Tensor:
        pass

    def _accumulator(self, X: torch.Tensor) -> torch.Tensor:
        return self.accumulator(X) if self.accumulator is not None else X

    def _classify_outlier(self, scores: torch.Tensor) -> torch.Tensor:
        # check threshold has has been inferred.
        return scores > self.threshold if self.threshold_inferred else None

    def _p_vals(self, scores: torch.Tensor) -> torch.Tensor:
        return (1 + (scores[:, None] < self.val_scores).sum(-1))/len(self.val_scores) \
            if self.threshold_inferred else None

    def infer_threshold(self, X: torch.Tensor, fpr: float) -> None:
        # check detector has been fit.
        self.val_scores = self.score(X)
        self.val_scores = self._accumulator(self.val_scores)
        self.threshold = torch.quantile(self.val_scores, 1-fpr)
        self.threshold_inferred = True

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        output = {'threshold_inferred': self.threshold_inferred}
        raw_scores = self.score(X)
        output['scores'] = self._accumulator(raw_scores)
        output['preds'] = self._classify_outlier(output['scores'])
        output['p_vals'] = self._p_vals(output['scores'])
        return output
