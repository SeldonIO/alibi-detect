import numpy as np


class BaseProcessor():
    def __init__(self, functions):
        self.functions = functions

    def __call__(self, X):
        return np.array([function(X) for function in self.functions])
