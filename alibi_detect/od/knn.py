from typing import Callable, Literal, Union, Optional
import numpy as np
import os

from alibi_detect.od.base import OutlierDetector
from alibi_detect.od.aggregation import BaseTransform

from alibi_detect.od.backends import KNNTorch, KNNKeops
from alibi_detect.utils.frameworks import BackendValidator
from alibi_detect.od.config import ConfigMixin
from alibi_detect.saving.registry import registry

X_REF_FILENAME = 'x_ref.npy'

backends = {
    'pytorch': KNNTorch,
    'keops': KNNKeops
}

@registry.register('KNN')
class KNN(OutlierDetector, ConfigMixin):
    CONFIG_PARAMS = ('k', 'kernel', 'aggregator', 'normaliser', 'backend')
    LARGE_PARAMS = ()
    BASE_OBJ = True

    def __init__(
        self,
        k: Union[int, np.ndarray],
        kernel: Optional[Callable] = None,
        aggregator: Union[BaseTransform, None] = None,
        normaliser: Union[BaseTransform, None] = None,
        backend: Literal['pytorch', 'keops'] = 'pytorch'
    ) -> None:
        print(locals())
        self._set_config(locals())
        backend = backend.lower()
        BackendValidator(
            backend_options={'pytorch': ['pytorch'],
                             'keops': ['keops']},
            construct_name=self.__class__.__name__
        ).verify_backend(backend)

        self.k = k
        self.kernel = kernel
        self.ensemble = isinstance(self.k, np.ndarray)
        self.normaliser = normaliser
        self.aggregator = aggregator
        self.fitted = False
        self.backend = backends[backend]

    def fit(self, X: np.ndarray) -> None:
        self.x_ref = self.backend.fit(X)
        val_scores = self.score(X)
        if getattr(self, 'normaliser'): self.normaliser.fit(val_scores)

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.backend.score(X, self.x_ref, self.k, kernel=self.kernel)

    # def _get_state(self):
    #     if self.kernel:
    #         kernel_state = self.kernel._get_state()

    #     return {
    #         'x_ref': X_REF_FILENAME,
    #         'kernel': kernel_state
    #     }
