from typing import Callable, Union, Optional, List, Dict, Any

from typing_extensions import Literal
import numpy as np

from alibi_detect.od.base import OutlierDetector, TransformProtocol, transform_protocols
from alibi_detect.od import backend as backend_objs
from alibi_detect.od.backend import normalizer_literals, aggregator_literals, KNNTorch, AccumulatorTorch
from alibi_detect.utils.frameworks import BackendValidator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


backends = {
    'pytorch': (KNNTorch, AccumulatorTorch)
}


class KNN(OutlierDetector):
    def __init__(
        self,
        k: Union[int, np.ndarray],
        kernel: Optional[Callable] = None,
        normalizer: Optional[Union[transform_protocols, normalizer_literals]] = 'ShiftAndScaleNormalizerTorch',
        aggregator: Union[TransformProtocol, aggregator_literals] = 'AverageAggregatorTorch',
        device: Optional[Union[Literal['cuda', 'gpu', 'cpu'], 'torch.device']] = None,
        backend: Literal['pytorch'] = 'pytorch',
    ) -> None:
        """
        k-Nearest Neighbours (kNN) outlier detector.

        Parameters
        ----------
        k
            Number of nearest neighbours to use for outlier detection. If an array is passed, an aggregator is required
            to aggregate the scores.
        kernel
            Kernel function to use for outlier detection. If None, `torch.cdist` is used.
        normalizer
            Normalizer to use for outlier detection. If None, no normalisation is applied.
        aggregator
            Aggregator to use for outlier detection. Can be set to None if `k` is a single value.
        backend
            Backend used for outlier detection. Defaults to `'pytorch'`. Options are `'pytorch'`.
        device
            Device type used. The default tries to use the GPU and falls back on CPU if needed. Can be specified by
            passing either 'cuda', 'gpu' or 'cpu'. Only relevant for 'pytorch' backend.

        Raises
        ------
        ValueError
            If `k` is an array and `aggregator` is None.
        NotImplementedError
            If choice of `backend` is not implemented.
        """
        super().__init__()

        backend_str: str = backend.lower()
        BackendValidator(
            backend_options={'pytorch': ['pytorch']},
            construct_name=self.__class__.__name__
        ).verify_backend(backend_str)

        backend_cls, accumulator_cls = backends[backend]
        accumulator = None
        if isinstance(k, (list, np.ndarray, tuple)):
            if isinstance(aggregator, str):
                aggregator = getattr(backend_objs, aggregator)()
            if aggregator is None:
                raise ValueError("If `k` is an array, an aggregator is required.")
            if isinstance(normalizer, str):
                normalizer = getattr(backend_objs, normalizer)()
            accumulator = accumulator_cls(
                normalizer=normalizer,
                aggregator=aggregator
            )
        self.backend = backend_cls(k, kernel=kernel, accumulator=accumulator, device=device)

    def fit(self, x_ref: Union[np.ndarray, List]) -> None:
        """Fit the kNN detector on reference data.

        Parameters
        ----------
        x_ref
            Reference data used to fit the kNN detector.
        """
        self.backend.fit(self.backend._to_tensor(x_ref))

    def score(self, x: Union[np.ndarray, List]) -> np.ndarray:
        """Score x instances using the kNN.

        Parameters
        ----------
        x
            Data to score. The shape of `x` should be `(n_instances, n_features)`.

        Returns
        -------
        Outlier scores. The shape of the scores is `(n_instances,)`. The higher the score, the more anomalous the \
        instance.
        """
        score = self.backend.score(self.backend._to_tensor(x))
        return self.backend._to_numpy(score)

    def infer_threshold(self, x_ref: Union[np.ndarray, List], fpr: float) -> None:
        """Infer the threshold for the kNN detector. The threshold is inferred using the reference data and the false
        positive rate. The threshold is used to determine the outlier labels in the predict method.

        Parameters
        ----------
        x_ref
            Reference data used to infer the threshold.
        fpr
            False positive rate used to infer the threshold. The false positive rate is the proportion of instances in \
            `x_ref` that are incorrectly classified as outliers. The false positive rate should be in the range \
            `(0, 1)`.
        """
        self.backend.infer_threshold(self.backend._to_tensor(x_ref), fpr)

    def predict(self, x: Union[np.ndarray, List]) -> Dict[str, Any]:
        """Predict whether the instances in x are outliers or not.

        Parameters
        ----------
        x
            Data to predict. The shape of `x` should be `(n_instances, n_features)`.

        Returns
        -------
        Dict with keys 'data' and 'meta'. 'data' contains the outlier scores. If threshold inference was performed, \
        'data' also contains the threshold value, outlier labels and p_vals . The shape of the scores is \
        `(n_instances,)`. The higher the score, the more anomalous the instance. 'meta' contains information about \
        the detector.
        """
        outputs = self.backend.predict(self.backend._to_tensor(x))
        output: Dict[str, Any] = {
            'data': outputs._to_numpy(),
            'meta': self.meta
        }
        return output
