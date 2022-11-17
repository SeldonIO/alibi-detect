import numpy as np
import torch


class KNNTorch(torch.nn.Module):
    def __init__(self, k, kernel=None):
        super().__init__()
        self.kernel = kernel
        self.ensemble = isinstance(k, (np.ndarray, list, tuple))
        self.ks = torch.tensor(k) if self.ensemble else torch.tensor([k])

    def forward(self, X):
        K = -self.kernel(X, self.x_ref) if self.kernel is not None else torch.cdist(X, self.x_ref)
        bot_k_dists = torch.topk(K, torch.max(self.ks), dim=1, largest=False)
        all_knn_dists = bot_k_dists.values[:, self.ks-1]
        all_knn_dists = all_knn_dists if self.ensemble else all_knn_dists[:, 0]
        return all_knn_dists.cpu()

    def fit(self, X: torch.tensor):
        self.x_ref = torch.as_tensor(X, dtype=torch.float32)
