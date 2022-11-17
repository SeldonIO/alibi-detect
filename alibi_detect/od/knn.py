from typing import Callable, Literal, Union, Optional
import numpy as np

from alibi_detect.od.base import OutlierDetector
from alibi_detect.od.backend.torch.ensemble import Accumulator

from alibi_detect.od.backend import KNNTorch
from alibi_detect.od.backend import KNNKeops
from alibi_detect.utils.frameworks import BackendValidator

X_REF_FILENAME = 'x_ref.npy'

backends = {
    'pytorch': KNNTorch,
    'keops': KNNKeops
}


class KNN(OutlierDetector):
    def __init__(
        self,
        k: Union[int, np.ndarray],
        kernel: Optional[Callable] = None,
        accumulator: Optional[Accumulator] = None,
        backend: Literal['pytorch', 'keops'] = 'pytorch'
    ) -> None:
        super().__init__()
        backend = backend.lower()
        BackendValidator(
            backend_options={'pytorch': ['pytorch'],
                             'keops': ['keops']},
            construct_name=self.__class__.__name__
        ).verify_backend(backend)

        self.k = k
        self.kernel = kernel
        self.ensemble = isinstance(self.k, (np.ndarray, list))
        self.accumulator = accumulator
        self.fitted = False
        self.backend = backends[backend]

    def fit(self, X: np.ndarray) -> None:
        self.x_ref = self.backend.fit(X)
        val_scores = self.score(X)
        if getattr(self, 'normaliser'):
            self.normaliser.fit(val_scores)

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.backend.score(X, self.x_ref, self.k, kernel=self.kernel)
