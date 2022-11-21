from __future__ import annotations
from abc import ABC, abstractmethod
import torch

import logging

logger = logging.getLogger(__name__)


class TorchOutlierDetector(torch.nn.Module, ABC):
    """ Base class for outlier detection algorithms. """
    threshold_inferred = False

    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def fit(self, X: torch.Tensor) -> None:
        pass

    @abstractmethod
    def score(self, X: torch.Tensor) -> torch.Tensor:
        pass

    def infer_threshold(self, X: torch.Tensor, fpr: float) -> None:
        """
        Infers the threshold above which only fpr% of inlying data scores.
        Also saves down the scores to be later used for computing p-values
            of new data points (by comparison to the empirical cdf).
        For ensemble models the scores are normalised and aggregated before
            saving scores and inferring threshold.
        """
        self.val_scores = self.score(X)
        self.val_scores = self.accumulator(self.val_scores) if self.accumulator is not None \
            else self.val_scores
        self.threshold = torch.quantile(self.val_scores, 1-fpr)
        self.threshold_inferred = True

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Scores the instances and then compares to pre-inferred threshold.
        For ensemble models the scores from each constituent is added to the output.
        p-values are also returned by comparison to validation scores (of inliers)
        """
        output = {}
        scores = self.score(X)
        output['raw_scores'] = scores

        if getattr(self, 'normaliser'):
            scores = self.normaliser.transform(scores)
            output['normalised_scores'] = scores

        if getattr(self, 'aggregator'):
            scores = self.aggregator.transform(scores)
            output['aggregate_scores'] = scores

        if self.threshold_inferred:
            p_vals = (1 + (scores[:, None] < self.val_scores).sum(-1))/len(self.val_scores)
            preds = scores > self.threshold
            output.update(scores=scores, preds=preds, p_vals=p_vals)

        return output
