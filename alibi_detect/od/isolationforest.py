import logging
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Dict, Union
from alibi_detect.base import BaseDetector, FitMixin, ThresholdMixin, outlier_prediction_dict

logger = logging.getLogger(__name__)


class IForest(BaseDetector, FitMixin, ThresholdMixin):

    def __init__(self,
                 threshold: float = None,
                 n_estimators: int = 100,
                 max_samples: Union[str, int, float] = 'auto',
                 max_features: Union[int, float] = 1.,
                 bootstrap: bool = False,
                 n_jobs: int = 1,
                 data_type: str = 'tabular'
                 ) -> None:
        """
        Outlier detector for tabular data using isolation forests.

        Parameters
        ----------
        threshold
            Threshold used for outlier score to determine outliers.
        n_estimators
            Number of base estimators in the ensemble.
        max_samples
            Number of samples to draw from the training data to train each base estimator.
            If int, draw 'max_samples' samples.
            If float, draw 'max_samples * number of features' samples.
            If 'auto', max_samples = min(256, number of samples)
        max_features
            Number of features to draw from the training data to train each base estimator.
            If int, draw 'max_features' features.
            If float, draw 'max_features * number of features' features.
        bootstrap
            Whether to fit individual trees on random subsets of the training data, sampled with replacement.
        n_jobs
            Number of jobs to run in parallel for 'fit' and 'predict'.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()

        if threshold is None:
            logger.warning('No threshold level set. Need to infer threshold using `infer_threshold`.')

        self.threshold = threshold
        self.isolationforest = IsolationForest(n_estimators=n_estimators,
                                               max_samples=max_samples,
                                               max_features=max_features,
                                               bootstrap=bootstrap,
                                               n_jobs=n_jobs)

        # set metadata
        self.meta['detector_type'] = 'offline'
        self.meta['data_type'] = data_type

    def fit(self,
            X: np.ndarray,
            sample_weight: np.ndarray = None
            ) -> None:
        """
        Fit isolation forest.

        Parameters
        ----------
        X
            Training batch.
        sample_weight
            Sample weights.
        """
        self.isolationforest.fit(X, sample_weight=sample_weight)

    def infer_threshold(self,
                        X: np.ndarray,
                        threshold_perc: float = 95.
                        ) -> None:
        """
        Update threshold by a value inferred from the percentage of instances considered to be
        outliers in a sample of the dataset.

        Parameters
        ----------
        X
            Batch of instances.
        threshold_perc
            Percentage of X considered to be normal based on the outlier score.
        """
        # compute outlier scores
        iscore = self.score(X)

        # update threshold
        self.threshold = np.percentile(iscore, threshold_perc)

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute outlier scores.

        Parameters
        ----------
        X
            Batch of instances to analyze.

        Returns
        -------
        Array with outlier scores for each instance in the batch.
        """
        return - self.isolationforest.decision_function(X)

    def predict(self,
                X: np.ndarray,
                return_instance_score: bool = True) \
            -> Dict[Dict[str, str], Dict[np.ndarray, np.ndarray]]:
        """
        Compute outlier scores and transform into outlier predictions.

        Parameters
        ----------
        X
            Batch of instances.
        return_instance_score
            Whether to return instance level outlier scores.

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the outlier predictions and instance level outlier scores.
        """
        # compute outlier scores
        iscore = self.score(X)

        # values above threshold are outliers
        outlier_pred = (iscore > self.threshold).astype(int)

        # populate output dict
        od = outlier_prediction_dict()
        od['meta'] = self.meta
        od['data']['is_outlier'] = outlier_pred
        if return_instance_score:
            od['data']['instance_score'] = iscore
        return od
