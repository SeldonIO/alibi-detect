from typing import Callable, Union, Optional, List, Dict
from typing_extensions import Literal
import numpy as np

from alibi_detect.od.base import OutlierDetector, TransformProtocol, FittedTransformProtocol
from alibi_detect.od.backend import KNNTorch, AccumulatorTorch
from alibi_detect.utils.frameworks import BackendValidator


backends = {
    'pytorch': (KNNTorch, AccumulatorTorch)
}


class KNN(OutlierDetector):
    def __init__(
        self,
        k: Union[int, np.ndarray],
        kernel: Optional[Callable] = None,
        normaliser: Optional[Union[TransformProtocol, FittedTransformProtocol]] = None,
        aggregator: Optional[TransformProtocol] = None,
        backend: Literal['pytorch'] = 'pytorch'
    ) -> None:
        super().__init__()

        backend_str: str = backend.lower()
        BackendValidator(
            backend_options={'pytorch': ['pytorch']},
            construct_name=self.__class__.__name__
        ).verify_backend(backend_str)

        if isinstance(k, (list, np.ndarray)) and aggregator is None:
            raise ValueError((f'k={k} is type {type(k)} but aggregator is {aggregator}, you must '
                              'specify at least an aggregator if you want to use the knn detector '
                              'ensemble like this.'))

        backend_cls, accumulator_cls = backends[backend]
        accumulator = None
        if normaliser is not None or aggregator is not None:
            accumulator = accumulator_cls(
                normaliser=normaliser,
                aggregator=aggregator
            )
        self.backend = backend_cls(k, kernel=kernel, accumulator=accumulator)

    def fit(self, x_ref: Union[np.ndarray, List]) -> None:
        self.backend.fit(self.backend._to_tensor(x_ref))

    def score(self, X: Union[np.ndarray, List]) -> np.ndarray:
        score = self.backend.score(self.backend._to_tensor(X))
        return score.numpy()

    def infer_threshold(self, x_ref: Union[np.ndarray, List], fpr: float) -> None:
        self.backend.infer_threshold(self.backend._to_tensor(x_ref), fpr)

    def predict(self, X: Union[np.ndarray, List]) -> Dict[str, np.ndarray]:
        outputs = self.backend.predict(self.backend._to_tensor(X))
        return outputs.numpy()
