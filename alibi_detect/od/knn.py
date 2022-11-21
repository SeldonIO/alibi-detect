from typing import Callable, Literal, Union, Optional
import numpy as np

from alibi_detect.od.base import OutlierDetector
from alibi_detect.od.backend.torch.ensemble import Accumulator

from alibi_detect.od.backend import KNNTorch
from alibi_detect.utils.frameworks import BackendValidator

X_REF_FILENAME = 'x_ref.npy'

backends = {
    'pytorch': KNNTorch,
}


class KNN(OutlierDetector):
    def __init__(
        self,
        k: Union[int, np.ndarray],
        kernel: Optional[Callable] = None,
        normaliser=None,
        aggregator=None,
        backend: Literal['pytorch'] = 'pytorch'
    ) -> None:
        super().__init__()
        backend = backend.lower()
        BackendValidator(
            backend_options={'pytorch': ['pytorch']},
            construct_name=self.__class__.__name__
        ).verify_backend(backend)

        # TODO: Abstract to pydantic model.
        if isinstance(k, (list, np.ndarray)) and aggregator is None:
            raise ValueError((f'k={k} is type {type(k)} but aggregator is {aggregator}, you must '
                              'specify at least an aggregator if you want to use the knn detector'
                              'an ensemble like this.'))

        self.k = k
        self.kernel = kernel
        self.ensemble = isinstance(self.k, (np.ndarray, list))
        self.normaliser = normaliser
        self.aggregator = aggregator
        self.fitted = False
        self.backend = backends[backend]

    def fit(self, X: np.ndarray) -> None:
        self.x_ref = self.backend.fit(X)
        val_scores = self.score(X)
        if getattr(self, 'normaliser'):
            self.normaliser.fit(val_scores)

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.backend.score(X, self.x_ref, self.k, kernel=self.kernel)
