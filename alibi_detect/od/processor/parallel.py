import ray
import numpy as np

from alibi_detect.od.processor.base import BaseProcessor
from alibi_detect.saving.registry import registry
from alibi_detect.od.config import ConfigMixin


@registry.register('ParallelProcessor')
class ParallelProcessor(BaseProcessor, ConfigMixin):
    def __init__(self):
        self._set_config(locals())

    def __call__(self, X, detectors):
        """This might not be the best way to do this. Need to checK:
        1. Does ray serlize each detector as we pass it to pfunc.
        2. Is this an issue?
        3. Compare runtimes against ListProcessor as a sanity check
        """
        @ray.remote
        def pfunc(obj, X):
            return obj.score(X)

        fn_refs = [pfunc.remote(detector, X) for detector in detectors]
        B, _ = X.shape
        results = np.empty((B, len(detectors)))
        for ind, fn_ref in enumerate(fn_refs):
            results[:, ind] = ray.get(fn_ref)
        return results
