from typing import List, Union, Optional, Dict
from dataclasses import dataclass, fields
from abc import ABC, abstractmethod

import numpy as np
import torch

from alibi_detect.od.pytorch.ensemble import FitMixinTorch
from alibi_detect.utils.pytorch.misc import get_device
from alibi_detect.exceptions import ThresholdNotInferredError
from alibi_detect.utils._types import TorchDeviceType


@dataclass
class TorchOutlierDetectorOutput:
    """Output of the outlier detector."""
    threshold_inferred: bool
    instance_score: torch.Tensor
    threshold: Optional[torch.Tensor]
    is_outlier: Optional[torch.Tensor]
    p_value: Optional[torch.Tensor]

    def to_frontend_dtype(self):
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, torch.Tensor):
                value = value.cpu().detach().numpy()
            if isinstance(value, np.ndarray) and value.ndim == 0:
                value = value.item()
            result[f.name] = value
        return result


def _tensor_to_frontend_dtype(x: Union[torch.Tensor, np.ndarray, float]) -> Union[np.ndarray, float]:
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    if isinstance(x, np.ndarray) and x.ndim == 0:
        x = x.item()
    return x  # type: ignore[return-value]


def _raise_type_error(x):
    raise TypeError(f'x is type={type(x)} but must be one of TorchOutlierDetectorOutput or a torch Tensor')


def to_frontend_dtype(x: Union[torch.Tensor, TorchOutlierDetectorOutput]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Converts any `torch` tensors found in input to `numpy` arrays.

    Takes a `torch` tensor or `TorchOutlierDetectorOutput` and converts any `torch` tensors found to `numpy` arrays

    Parameters
    ----------
    x
        Data to convert.

    Returns
    -------
    `np.ndarray` or dictionary of containing `numpy` arrays
    """

    return {
        'TorchOutlierDetectorOutput': lambda x: x.to_frontend_dtype(),
        'Tensor': _tensor_to_frontend_dtype
    }.get(
        x.__class__.__name__,
        _raise_type_error
    )(x)


class TorchOutlierDetector(torch.nn.Module, FitMixinTorch, ABC):
    """Base class for torch backend outlier detection algorithms."""
    threshold_inferred = False
    threshold = None

    def __init__(self, device: TorchDeviceType = None):
        self.device = get_device(device)
        super().__init__()

    @abstractmethod
    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Score the data.

        Parameters
        ----------
        x
            Data to score.

        """
        pass

    @torch.jit.unused
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
                arg: Union[torch.Tensor, TorchOutlierDetectorOutput]
            ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Converts input to frontend data format.

        This is an interface method that ensures that the output of the outlier detector is in a common format for
        different backends. Mostly this means converting `torch.tensors` to `np.ndarray`. If `arg` is a
        `TorchOutlierDetectorOutput` object, we call its `to_frontend_dtype` method. Otherwise, if `arg` is a
        `torch.Tensor`, we convert it to a `numpy` array.

        Parameters
        ----------
        x
            Data to convert.

        Returns
        -------
        `np.ndarray` or dictionary of containing `numpy` arrays
        """
        return to_frontend_dtype(arg)

    def _to_backend_dtype(self, x: Union[List, np.ndarray]) -> torch.Tensor:
        """Converts data from the frontend to the backend format.

        This is an interface method that ensures that the input of the chosen outlier detector backend is in the correct
        format.

        Parameters
        ----------
        x
            Data to convert.
        """
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _ensembler(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregates and normalizes the data

        If the detector has an ensembler attribute we use it to aggregate and normalize the data.

        Parameters
        ----------
        x
            Data to aggregate and normalize.

        Returns
        -------
        `torch.Tensor` or original data without alteration

        Raises
        ------
        ThresholdNotInferredError
            If the detector is an ensemble, and the ensembler used to aggregate the outlier scores has a fittable
            component, then the detector threshold must be inferred before predictions can be made. This is because
            while the scoring functionality of the detector is fit within the `.fit` method on the training data
            the ensembler has to be fit on the validation data along with the threshold and this is done in the
            `.infer_threshold` method.
        """
        if hasattr(self, 'ensembler') and self.ensembler is not None:
            # `type: ignore` here because self.ensembler here causes an error with mypy when using torch.jit.script.
            # For some reason it thinks self.ensembler is a torch.Tensor and therefore is not callable.
            if not torch.jit.is_scripting():
                if not self.ensembler.fitted:  # type: ignore
                    self.check_threshold_inferred()
            return self.ensembler(x)  # type: ignore
        else:
            return x

    def _classify_outlier(self, scores: torch.Tensor) -> torch.Tensor:
        """Classify the data as outlier or not.

        Parameters
        ----------
        scores
            Scores to classify. Larger scores indicate more likely outliers.

        Returns
        -------
        `torch.Tensor` or ``None``
        """
        return (scores > self.threshold).to(torch.int8) if self.threshold_inferred else None

    def _p_vals(self, scores: torch.Tensor) -> torch.Tensor:
        """Compute p-values for the scores.

        Parameters
        ----------
        scores
            Scores to compute p-values for.

        Returns
        -------
        `torch.Tensor` or ``None``
        """
        return (1 + (scores[:, None] < self.val_scores).sum(-1))/len(self.val_scores) \
            if self.threshold_inferred else None

    def infer_threshold(self, x: torch.Tensor, fpr: float):
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
        if self.ensemble:
            self.val_scores = self.ensembler.fit(self.val_scores).transform(self.val_scores)  # type: ignore
        self.threshold = torch.quantile(self.val_scores, 1-fpr, interpolation='higher')
        self.threshold_inferred = True

    def predict(self, x: torch.Tensor) -> TorchOutlierDetectorOutput:
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
        Output of the outlier detector. Includes the p-values, outlier labels, instance scores and threshold.

        Raises
        ------
        ValueError
            Raised if the detector is not fit on reference data.
        """
        self.check_fitted()
        raw_scores = self.score(x)
        scores = self._ensembler(raw_scores)

        return TorchOutlierDetectorOutput(
            instance_score=scores,
            is_outlier=self._classify_outlier(scores),
            p_value=self._p_vals(scores),
            threshold_inferred=self.threshold_inferred,
            threshold=self.threshold
        )
