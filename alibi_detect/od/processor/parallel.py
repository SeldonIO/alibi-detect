import ray
from alibi_detect.od.processor.base import BaseProcessor


class ParallelProcessor(BaseProcessor):
    def __init__(self, functions):
        self.functions = [ray.remote(function) for function in functions]
    
    def __call__(self, X):
        fn_refs = []
        for function in self.functions:
            fn_ref = function.remote(X)
            fn_refs.append(fn_ref)

        return [ray.get(fn_ref) for fn_ref in fn_refs]
