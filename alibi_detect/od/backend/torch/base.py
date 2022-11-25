from __future__ import annotations
import numpy as np
from typing import List, Dict, Union, Optional
import torch
from alibi_detect.od.backend.torch.ensemble import FitMixinTorch
from dataclasses import dataclass, asdict

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class TorchOutlierDetectorOutput:
    threshold_inferred: bool
    scores: torch.Tensor
    threshold: Optional[torch.Tensor]
    preds: Optional[torch.Tensor]
    p_vals: Optional[torch.Tensor]

    def numpy(self) -> Dict[str, Union[bool, Optional[torch.Tensor]]]:
        outputs = asdict(self)
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                outputs[key] = value.cpu().detach().numpy()
        return outputs


class TorchOutlierDetector(torch.nn.Module, FitMixinTorch, ABC):
    """ Base class for torch backend outlier detection algorithms."""
    threshold_inferred = False
    threshold = None

    def __init__(self):
        super().__init__()

    @abstractmethod
    def _fit(self, x_ref: torch.Tensor) -> None:
        raise NotImplementedError()

    @abstractmethod
    def score(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @torch.jit.ignore
    def check_threshould_infered(self):
        if not self.threshold_inferred:
            raise ValueError((f'{self.__class__.__name__} has no threshold set, '
                              'call `infer_threshold` before predicting.'))

    def _to_tensor(self, X: Union[List, np.ndarray]):
        return torch.as_tensor(X, dtype=torch.float32)

    def _accumulator(self, X: torch.Tensor) -> torch.Tensor:
        return self.accumulator(X) if self.accumulator is not None else X  # type: ignore

    def _classify_outlier(self, scores: torch.Tensor) -> torch.Tensor:
        return scores > self.threshold if self.threshold_inferred else None

    def _p_vals(self, scores: torch.Tensor) -> torch.Tensor:
        return (1 + (scores[:, None] < self.val_scores).sum(-1))/len(self.val_scores) \
            if self.threshold_inferred else None

    def infer_threshold(self, X: torch.Tensor, fpr: float) -> None:
        self.val_scores = self.score(X)
        self.val_scores = self._accumulator(self.val_scores)
        self.threshold = torch.quantile(self.val_scores, 1-fpr)
        self.threshold_inferred = True

    def predict(self, X: torch.Tensor) -> TorchOutlierDetectorOutput:
        self.check_fitted()  # type: ignore
        raw_scores = self.score(X)
        scores = self._accumulator(raw_scores)
        return TorchOutlierDetectorOutput(
            scores=scores,
            preds=self._classify_outlier(scores),
            p_vals=self._p_vals(scores),
            threshold_inferred=self.threshold_inferred,
            threshold=self.threshold
        )
