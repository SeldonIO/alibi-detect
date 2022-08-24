import numpy as np


class BaseProcessor():
    def __init__(self, detectors):
        self.detectors = detectors

    def __call__(self, X):
        B, _ = X.shape
        results = np.empty((B, len(self.detectors)))
        for ind, detector in enumerate(self.detectors):
            results[:, ind] = detector.score(X)
        return results
