import numpy as np
from alibi_detect.od.base import ConfigMixin
from alibi_detect.saving.registry import registry


@registry.register('BaseProcessor')
class BaseProcessor(ConfigMixin):
    def __init__(self):
        self._set_config(locals())

    def __call__(self, X, detectors):
        B, _ = X.shape
        results = np.empty((B, len(detectors)))
        for ind, detector in enumerate(detectors):
            results[:, ind] = detector.score(X)
        return results
