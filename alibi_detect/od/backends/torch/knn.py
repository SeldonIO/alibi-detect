import numpy as np
import torch


class KNNTorch:
    def score(X, x_ref, k, kernel=None):
        ensemble = isinstance(k, (np.ndarray, list, tuple))
        X = torch.as_tensor(X, dtype=torch.float32)
        K = -kernel(X, x_ref) if kernel else torch.cdist(X, x_ref)
        ks = np.array(k) if ensemble else np.array([k])
        bot_k_dists = torch.topk(K, np.max(ks), dim=1, largest=False)
        all_knn_dists = bot_k_dists.values[:,ks-1]
        all_knn_dists = all_knn_dists if ensemble else all_knn_dists[:,0]
        return all_knn_dists.cpu().numpy()

    def fit(X):
        return torch.as_tensor(X, dtype=torch.float32)
