from typing import List, Union, Optional, Dict
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from typing_extensions import Self

import numpy as np

from alibi_detect.exceptions import NotFittedError, ThresholdNotInferredError


@dataclass
class SklearnOutlierDetectorOutput:
    """Output of the outlier detector."""
    threshold_inferred: bool
    instance_score: np.ndarray
    threshold: Optional[np.ndarray]
    is_outlier: Optional[np.ndarray]
    p_value: Optional[np.ndarray]


class FitMixinSklearn(ABC):
    fitted = False

    @abstractmethod
    def fit(self, x_ref: np.ndarray) -> Self:
        """Abstract fit method.

        Parameters
        ----------
        x
            `torch.Tensor` to fit object on.
        """
        return self

    def _set_fitted(self) -> Self:
        """Sets the fitted attribute to True.

        Should be called within the object fit method.
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
        if not self.fitted:
            raise NotFittedError(self.__class__.__name__)


class SklearnOutlierDetector(FitMixinSklearn, ABC):
    """Base class for sklearn backend outlier detection algorithms."""
    threshold_inferred = False
    threshold = None

    @abstractmethod
    def score(self, x: np.ndarray) -> np.ndarray:
        """Score the data.

        Parameters
        ----------
        x
            Data to score.

        """
        pass

    def check_threshold_inferred(self):
        """Check if threshold is inferred.

        Raises
        ------
        ThresholdNotInferredError
            Raised if threshold is not inferred.
        """
        if not self.threshold_inferred:
            raise ThresholdNotInferredError(self.__class__.__name__)

    @staticmethod
    def _to_frontend_dtype(
            arg: Union[np.ndarray, SklearnOutlierDetectorOutput]
            ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Converts input to frontend data format.

        This is an interface method that ensures that the output of the outlier detector is in a common format for
        different backends. If `arg` is a `SklearnOutlierDetectorOutput` object, we unpack it into a `dict` and
        return it.

        Parameters
        ----------
        x
            Data to convert.

        Returns
        -------
        `np.ndarray` or dictionary containing frontend compatible data.
        """
        if isinstance(arg, SklearnOutlierDetectorOutput):
            return asdict(arg)
        return arg

    @staticmethod
    def _to_backend_dtype(x: Union[List, np.ndarray]) -> np.ndarray:
        """Converts data from the frontend to the backend format.

        This is an interface method that ensures that the input of the chosen outlier detector backend is in the correct
        format. In the case of the Sklearn backend, we ensure the data is a numpy array.

        Parameters
        ----------
        x
            Data to convert.
        """
        return np.asarray(x)

    def _classify_outlier(self, scores: np.ndarray) -> Optional[np.ndarray]:
        """Classify the data as outlier or not.

        Parameters
        ----------
        scores
            Scores to classify. Larger scores indicate more likely outliers.

        Returns
        -------
        `np.ndarray` or ``None``
        """
        if (self.threshold_inferred and self.threshold is not None):
            return (scores > self.threshold).astype(int)
        return None

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
        ValueError
            Raised if `fpr` is less than ``1/len(x)``.
        """
        if not 0 < fpr < 1:
            raise ValueError('`fpr` must be in `(0, 1)`.')
        if fpr < 1/len(x):
            raise ValueError(f'`fpr` must be greater than `1/len(x)={1/len(x)}`.')
        self.val_scores = self.score(x)
        self.threshold = np.quantile(self.val_scores, 1-fpr, interpolation='higher')  # type: ignore[call-overload]
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

        Returns
        -------
        `SklearnOutlierDetectorOutput`
            Output of the outlier detector.

        Raises
        ------
        ValueError
            Raised if the detector is not fit on reference data.
        """
        self.check_fitted()
        scores = self.score(x)

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
        scores = self.score(x)
        self.check_threshold_inferred()
        return self._classify_outlier(scores)
