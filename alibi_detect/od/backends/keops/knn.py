import numpy as np
import torch
from pykeops.torch import LazyTensor


def cdist(X, Y):
    return ((X - Y)**2).sum(-1).sqrt()


class KnnKeops:
    def score(X, x_ref, k, kernel=None):
        ensemble = isinstance(k, (np.ndarray, list, tuple))
        X = torch.as_tensor(X)
        X_keops = LazyTensor(X[:, None, :])
        x_ref_keops = LazyTensor(x_ref[None, :, :])
        K = -kernel(X_keops, x_ref_keops) if kernel else cdist(X_keops, x_ref_keops)
        ks = np.array(k) if ensemble else np.array([k])
        bot_k_inds = K.argKmin(np.max(ks), dim=1)
        all_knn_dists = (X[:, None, :] - x_ref[bot_k_inds][:, ks-1, :]).norm(dim=2)
        all_knn_dists = all_knn_dists if ensemble else all_knn_dists[:,0]
        return all_knn_dists.cpu().numpy()


    def fit(X):
        return torch.as_tensor(X)