import ray
import numpy as np

from alibi_detect.od.processor.base import BaseProcessor


class ParallelProcessor(BaseProcessor):
    def __init__(self, detectors):
        self.detectors = detectors
    
    def __call__(self, X):
        """This might not be the best way to do this. Need to checK:
        1. Does ray serlize each detector as we pass it to pfunc.
        2. Is this an issue?
        3. Compare runtimes against ListProcessor as a sanity check
        """
        @ray.remote
        def pfunc(obj, X):
            return obj.score(X)

        fn_refs = [pfunc.remote(detector, X) for detector in self.detectors]
        B, _ = X.shape
        results = np.empty((B, len(self.detectors)))
        for ind, fn_ref in enumerate(fn_refs):
            results[:, ind] = ray.get(fn_ref)
        return results
