from __future__ import annotations
from typing import List, Dict, Union, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

import numpy as np
import torch

from alibi_detect.od.backend.torch.ensemble import FitMixinTorch
from alibi_detect.utils.pytorch.misc import get_device


@dataclass
class TorchOutlierDetectorOutput:
    """Output of the outlier detector."""
    threshold_inferred: bool
    scores: torch.Tensor
    threshold: Optional[torch.Tensor]
    preds: Optional[torch.Tensor]
    p_vals: Optional[torch.Tensor]

    def _to_numpy(self) -> Dict[str, Union[bool, Optional[torch.Tensor]]]:
        """Converts the output to numpy."""
        outputs = asdict(self)
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                outputs[key] = value.cpu().detach().numpy()
        return outputs


class TorchOutlierDetector(torch.nn.Module, FitMixinTorch, ABC):
    """Base class for torch backend outlier detection algorithms."""
    threshold_inferred = False
    threshold = None

    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        self.device = get_device(device)
        super().__init__()

    @abstractmethod
    def _fit(self, x_ref: torch.Tensor) -> None:
        """Fit the outlier detector to the reference data.

        Parameters
        ----------
        x_ref
            Reference data.

        Raises
        ------
        `NotImplementedError`
            Raised if not implemented.
        """
        raise NotImplementedError()

    @abstractmethod
    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Score the data.

        Parameters
        ----------
        x
            Data to score.

        Raises
        ------
        `NotImplementedError`
            Raised if not implemented.
        """
        raise NotImplementedError()

    @torch.jit.unused
    def check_threshould_infered(self):
        """Check if threshold is inferred.

        Raises
        ------
        ValueError
            Raised if threshold is not inferred.
        """
        if not self.threshold_inferred:
            raise ValueError((f'{self.__class__.__name__} has no threshold set, '
                              'call `infer_threshold` before predicting.'))

    def _to_tensor(self, x: Union[List, np.ndarray]):
        """Converts the data to a tensor.

        Parameters
        ----------
        x
            Data to convert.

        Returns
        -------
        `torch.Tensor`
        """
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _to_numpy(self, x: torch.Tensor):
        """Converts the data to numpy.

        Parameters
        ----------
        x
            Data to convert.

        Returns
        -------
        `np.ndarray`
        """
        return x.cpu().detach().numpy()

    def _accumulator(self, x: torch.Tensor) -> torch.Tensor:
        """Accumulates the data.

        Parameters
        ----------
        x
            Data to accumulate.

        Returns
        -------
        `torch.Tensor` or just returns original data
        """
        # `type: ignore` here becuase self.accumulator here causes an error with mypy when using torch.jit.script.
        # For some reason it thinks self.accumulator is a torch.Tensor and therefore is not callable.
        return self.accumulator(x) if self.accumulator is not None else x  # type: ignore

    def _classify_outlier(self, scores: torch.Tensor) -> torch.Tensor:
        """Classify the data as outlier or not.

        Parameters
        ----------
        scores
            Scores to classify. Larger scores indicate more likely outliers.

        Returns
        -------
        `torch.Tensor` or `None`
        """
        return scores > self.threshold if self.threshold_inferred else None

    def _p_vals(self, scores: torch.Tensor) -> torch.Tensor:
        """Compute p-values for the scores.

        Parameters
        ----------
        scores
            Scores to compute p-values for.

        Returns
        -------
        `torch.Tensor` or `None`
        """
        return (1 + (scores[:, None] < self.val_scores).sum(-1))/len(self.val_scores) \
            if self.threshold_inferred else None

    def infer_threshold(self, x: torch.Tensor, fpr: float) -> None:
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
            Raised if fpr is not in (0, 1).
        """
        if not 0 < fpr < 1:
            ValueError('fpr must be in (0, 1).')
        self.val_scores = self.score(x)
        self.val_scores = self._accumulator(self.val_scores)
        self.threshold = torch.quantile(self.val_scores, 1-fpr)
        self.threshold_inferred = True

    def predict(self, x: torch.Tensor) -> TorchOutlierDetectorOutput:
        """Predict outlier labels for the data.

        Computes the outlier scores. If the detector is not fit on reference data we raise an error.
        If the threshold is inferred, the outlier labels and p-values are also computed and returned.
        Otherwise, the outlier labels and p-values are set to `None`.

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
        `TorchOutlierDetectorOutput`
            Output of the outlier detector.

        """
        self.check_fitted()  # type: ignore
        raw_scores = self.score(x)
        scores = self._accumulator(raw_scores)
        return TorchOutlierDetectorOutput(
            scores=scores,
            preds=self._classify_outlier(scores),
            p_vals=self._p_vals(scores),
            threshold_inferred=self.threshold_inferred,
            threshold=self.threshold
        )
