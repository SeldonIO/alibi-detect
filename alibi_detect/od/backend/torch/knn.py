from typing import Optional
import numpy as np
import torch
from alibi_detect.od.backend.torch.ensemble import Accumulator
from alibi_detect.od.backend.torch.base import TorchOutlierDetector


class KNNTorch(TorchOutlierDetector):
    def __init__(
            self,
            k,
            kernel=None,
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
        preds = self._classify_outlier(scores)
        return preds.cpu() if preds is not None else scores

    def score(self, X):
        K = -self.kernel(X, self.x_ref) if self.kernel is not None else torch.cdist(X, self.x_ref)
        bot_k_dists = torch.topk(K, torch.max(self.ks), dim=1, largest=False)
        all_knn_dists = bot_k_dists.values[:, self.ks-1]
        return all_knn_dists if self.ensemble else all_knn_dists[:, 0]

    def fit(self, X: torch.tensor):
        self.x_ref = torch.as_tensor(X, dtype=torch.float32)
        if self.accumulator is not None:
            scores = self.score(X)
            self.accumulator.fit(scores)
