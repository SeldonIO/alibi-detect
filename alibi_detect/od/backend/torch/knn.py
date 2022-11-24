from typing import Optional, Union, List
import numpy as np
import torch
from alibi_detect.od.backend.torch.ensemble import Accumulator
from alibi_detect.od.backend.torch.base import TorchOutlierDetector


class KNNTorch(TorchOutlierDetector):
    def __init__(
            self,
            k: Union[np.ndarray, List],
            kernel: Optional[torch.nn.Module] = None,
            accumulator: Optional[Accumulator] = None
            ):
        super().__init__()
        self.kernel = kernel
        self.ensemble = isinstance(k, (np.ndarray, list, tuple))
        self.ks = torch.tensor(k) if self.ensemble else torch.tensor([k])
        self.accumulator = accumulator

    def forward(self, X):
        raw_scores = self.score(X)
        scores = self._accumulator(raw_scores)
        self.check_threshould_infered()
        preds = scores > self.threshold
        return preds.cpu()

    def score(self, X):
        self.check_fitted()
        K = -self.kernel(X, self.x_ref) if self.kernel is not None else torch.cdist(X, self.x_ref)
        bot_k_dists = torch.topk(K, torch.max(self.ks), dim=1, largest=False)
        all_knn_dists = bot_k_dists.values[:, self.ks-1]
        return all_knn_dists if self.ensemble else all_knn_dists[:, 0]

    def _fit(self, x_ref: torch.tensor):
        self.x_ref = x_ref
        if self.accumulator is not None:
            scores = self.score(x_ref)
            self.accumulator.fit(scores)
