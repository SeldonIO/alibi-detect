from __future__ import annotations
from typing import List, Union, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

import numpy as np

from alibi_detect.od.base import NotFitException
from alibi_detect.od.base import ThresholdNotInferredException


@dataclass
class SklearnOutlierDetectorOutput:
    """Output of the outlier detector."""
    threshold_inferred: bool
    instance_score: np.ndarray
    threshold: Optional[np.ndarray]
    is_outlier: Optional[np.ndarray]
    p_value: Optional[np.ndarray]


class FitMixin(ABC):
    _fitted = False

    def __init__(self):
        """Fit mixin

        Utility class that provides fitted checks for alibi-detect objects that require to be fit before use.
        """
        super().__init__()

    def fit(self, x: np.ndarray) -> FitMixin:
        self._fitted = True
        self._fit(x)
        return self

    @abstractmethod
    def _fit(self, x: np.ndarray):
        """Fit on `x` array.

        This method should be overidden on child classes.

        Parameters
        ----------
        x
            Reference `np.array` for fitting object.
        """
        pass

    def check_fitted(self):
        """Raises error if parent object instance has not been fit.

        Raises
        ------
        NotFitException
            Raised if method called and object has not been fit.
        """
        if not self._fitted:
            raise NotFitException(f'{self.__class__.__name__} has not been fit!')


class SklearnOutlierDetector(FitMixin, ABC):
    """Base class for sklearn backend outlier detection algorithms."""
    threshold_inferred = False
    threshold = None

    def __init__(self):
        super().__init__()

    @abstractmethod
    def _fit(self, x_ref: np.ndarray) -> None:
        """Fit the outlier detector to the reference data.

        Parameters
        ----------
        x_ref
            Reference data.
        """
        pass

    @abstractmethod
    def score(self, x: np.ndarray) -> np.ndarray:
        """Score the data.

        Parameters
        ----------
        x
            Data to score.

        """
        pass

    def check_threshold_infered(self):
        """Check if threshold is inferred.

        Raises
        ------
        ThresholdNotInferredException
            Raised if threshold is not inferred.
        """
        if not self.threshold_inferred:
            raise ThresholdNotInferredException((f'{self.__class__.__name__} has no threshold set, '
                                                 'call `infer_threshold` before predicting.'))

    @staticmethod
    def _to_numpy(arg):
        """Map params to numpy arrays.

        This function is for interface compatibility with the other backends. As such it does nothing but
        return the input.

        Parameters
        ----------
        x
            Data to convert.

        Returns
        -------
        `np.ndarray` or dictionary of containing `numpy` arrays
        """
        if isinstance(arg, SklearnOutlierDetectorOutput):
            return asdict(arg)
        return arg

    @staticmethod
    def _to_tensor(x: Union[List, np.ndarray]) -> np.ndarray:
        """Converts the data to a tensor.

        This function is for interface compatibility with the other backends. As such it does nothing but
        return the input.

        Parameters
        ----------
        x
            Data to convert.

        Returns
        -------
        `np.ndarray`
        """
        return np.array(x)

    def _ensembler(self, x: np.ndarray) -> np.ndarray:
        """Aggregates and normalizes the data

        If the detector has an ensembler attribute we use it to aggregate and normalize the data.

        Parameters
        ----------
        x
            Data to aggregate and normalize.

        Returns
        -------
        `np.ndarray` or just returns original data
        """
        if hasattr(self, 'ensembler') and self.ensembler is not None:
            return self.ensembler(x)
        else:
            return x

    def _classify_outlier(self, scores: np.ndarray) -> np.ndarray:
        """Classify the data as outlier or not.

        Parameters
        ----------
        scores
            Scores to classify. Larger scores indicate more likely outliers.

        Returns
        -------
        `np.ndarray` or ``None``
        """
        return scores > self.threshold if self.threshold_inferred else None

    def _p_vals(self, scores: np.ndarray) -> np.ndarray:
        """Compute p-values for the scores.

        Parameters
        ----------
        scores
            Scores to compute p-values for.

        Returns
        -------
        `np.ndarray` or ``None``
        """
        return (1 + (scores[:, None] < self.val_scores).sum(-1))/len(self.val_scores) \
            if self.threshold_inferred else None

    def infer_threshold(self, x: np.ndarray, fpr: float) -> None:
        """Infer the threshold for the data. Prerequisite for outlier predictions.

        Parameters
        ----------
        x
            Data to infer the threshold for.
        fpr
            False positive rate to use for threshold inference.

        Raises
        ------
        ValueError
            Raised if `fpr` is not in ``(0, 1)``.
        """
        if not 0 < fpr < 1:
            ValueError('`fpr` must be in `(0, 1)`.')
        self.val_scores = self.score(x)
        self.val_scores = self._ensembler(self.val_scores)
        self.threshold = np.quantile(self.val_scores, 1-fpr)
        self.threshold_inferred = True

    def predict(self, x: np.ndarray) -> SklearnOutlierDetectorOutput:
        """Predict outlier labels for the data.

        Computes the outlier scores. If the detector is not fit on reference data we raise an error.
        If the threshold is inferred, the outlier labels and p-values are also computed and returned.
        Otherwise, the outlier labels and p-values are set to ``None``.

        Parameters
        ----------
        x
            Data to predict.

        Raises
        ------
        ValueError
            Raised if the detector is not fit on reference data.

        Returns
        -------
        `SklearnOutlierDetectorOutput`
            Output of the outlier detector.

        """
        self.check_fitted()  # type: ignore
        raw_scores = self.score(x)
        scores = self._ensembler(raw_scores)

        return SklearnOutlierDetectorOutput(
            instance_score=scores,
            is_outlier=self._classify_outlier(scores),
            p_value=self._p_vals(scores),
            threshold_inferred=self.threshold_inferred,
            threshold=self.threshold
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Classify outliers.

        Parameters
        ----------
        x
            Data to classify.
        """
        raw_scores = self.score(x)
        scores = self._ensembler(raw_scores)
        self.check_threshold_infered()
        return self._classify_outlier(scores)
