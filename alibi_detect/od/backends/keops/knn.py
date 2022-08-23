import numpy as np
import torch
from pykeops.torch import LazyTensor



class KnnKeops:
    def score(X, x_ref, k, kernel=None):
        ensemble = len(k) > 1
        X = torch.as_tensor(X)
        X = LazyTensor(X[:, :, None, :])
        x_ref = LazyTensor(x_ref[:, None, :, :])
        K = -kernel(X, x_ref) if kernel else torch.cdist(X, x_ref)
        ks = k if ensemble else np.array([k])
        bot_k_dists = torch.topk(K, np.max(ks), dim=1, largest=False)
        all_knn_dists = bot_k_dists.values[:,ks-1]
        all_knn_dists = all_knn_dists if ensemble else all_knn_dists[:,0]
        return all_knn_dists.cpu().numpy()


    def fit(X):
        return torch.as_tensor(X)