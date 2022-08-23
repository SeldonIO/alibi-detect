from typing import Callable, Literal, Union, Optional
import numpy as np
# import torch

from alibi_detect.od.base import OutlierDetector
from alibi_detect.od.aggregation import AverageAggregator, MaxAggregator, MinAggregator, TopKAggregator
from alibi_detect.od.aggregation import ShiftAndScaleNormaliser, PValNormaliser

from alibi_detect.od.backends import KnnTorch, KnnKeops
from alibi_detect.utils.frameworks import BackendValidator

backends = {
    'pytorch': KnnTorch,
    'keops': KnnKeops
}


class KNN(OutlierDetector):
    def __init__(
        self,
        k: Union[int, np.ndarray],
        kernel: Optional[Callable] = None,
        aggregator: Union[AverageAggregator, MaxAggregator, MinAggregator, TopKAggregator, None] = None,
        normaliser: Union[ShiftAndScaleNormaliser, PValNormaliser, None] = None,
        backend: Literal['pytorch', 'keops'] = 'pytorch'
    ) -> None:

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
