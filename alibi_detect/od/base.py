from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from alibi_detect.version import __version__
import logging
from alibi_detect.base import BaseDetector

logger = logging.getLogger(__name__)


class OutlierDetector(BaseDetector, ABC):
    """ Base class for outlier detection algorithms. """
    threshold_inferred = False

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        pass


    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        pass


    def infer_threshold(self, X: np.ndarray, fpr: float) -> None:
        """
        Infers the threshold above which only fpr% of inlying data scores.
        Also saves down the scores to be later used for computing p-values
            of new data points (by comparison to the empirical cdf).
        For ensemble models the scores are normalised and aggregated before
            saving scores and inferring threshold.
        """
        self.val_scores = self.score(X)
        self.val_scores = self.normaliser.fit(self.val_scores).transform(self.val_scores) \
            if getattr(self, 'normaliser') else self.val_scores
        self.val_scores = self.aggregator.fit(self.val_scores).transform(self.val_scores) \
            if getattr(self, 'aggregator') else self.val_scores
        self.threshold = np.quantile(self.val_scores, 1-fpr)
        self.threshold_inferred = True


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Scores the instances and then compares to pre-inferred threshold.
        For ensemble models the scores from each constituent is added to the output.
        p-values are also returned by comparison to validation scores (of inliers)
        """
        output = {}
        scores = self.score(X)
        output['raw_scores'] = scores

        if getattr(self, 'normaliser') and self.normaliser.fitted:
            scores = self.normaliser.transform(scores)
            output['normalised_scores'] = scores

        if getattr(self, 'aggregator') and self.aggregator.fitted:
            scores = self.aggregator.transform(scores)
            output['aggregate_scores'] = scores

        if self.threshold_inferred:
            p_vals = (1 + (scores[:, None] < self.val_scores).sum(-1))/len(self.val_scores)
            preds = scores > self.threshold
            output.update(scores=scores, preds=preds, p_vals=p_vals)

        return output