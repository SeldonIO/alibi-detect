from typing import Callable, Union, Optional, List, Dict, Any

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
        normalizer: Optional[Union[TransformProtocol, FittedTransformProtocol]] = None,
        aggregator: Optional[TransformProtocol] = None,
        backend: Literal['pytorch'] = 'pytorch'
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
            Aggregator to use for outlier detection. If None, no aggregation is applied. If an array is passed for `k`,
            then an aggregator is required.
        backend
            Backend used for outlier detection. Defaults to `'pytorch'`. Options are `'pytorch'`.

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

        if isinstance(k, (list, np.ndarray)) and aggregator is None:
            raise ValueError((f'k={k} is type {type(k)} but aggregator is {aggregator}, you must '
                              'specify at least an aggregator if you want to use the knn detector '
                              'ensemble like this.'))

        backend_cls, accumulator_cls = backends[backend]
        accumulator = None
        if normalizer is not None or aggregator is not None:
            accumulator = accumulator_cls(
                normalizer=normalizer,
                aggregator=aggregator
            )
        self.backend = backend_cls(k, kernel=kernel, accumulator=accumulator)

    def fit(self, x_ref: Union[np.ndarray, List]) -> None:
        """Fit the kNN detector on reference data.

        Parameters
        ----------
        x_ref
            Reference data used to fit the kNN detector.
        """
        self.backend.fit(self.backend._to_tensor(x_ref))

    def score(self, X: Union[np.ndarray, List]) -> np.ndarray:
        """Score X instances using the kNN.

        Parameters
        ----------
        X
            Data to score. The shape of `X` should be `(n_instances, n_features)`.

        Returns
        -------
        Anomaly scores. The shape of the scores is `(n_instances,)`. The higher the score, the more anomalous the
        instance.
        """
        score = self.backend.score(self.backend._to_tensor(X))
        return score.numpy()

    def infer_threshold(self, x_ref: Union[np.ndarray, List], fpr: float) -> None:
        """Infer the threshold for the kNN detector.

        Parameters
        ----------
        x_ref
            Reference data used to infer the threshold.
        fpr
            False positive rate used to infer the threshold.
        """
        self.backend.infer_threshold(self.backend._to_tensor(x_ref), fpr)

    def predict(self, X: Union[np.ndarray, List]) -> Dict[str, Any]:
        """Predict whether the instances in X are outliers or not.

        Parameters
        ----------
        X
            Data to predict. The shape of `X` should be `(n_instances, n_features)`.

        Returns
        -------
        Dict with keys 'data' and 'meta'. 'data' contains the outlier scores. If threshold inference was performed,
        'data' also contains the threshold value, outlier labels and p_vals . The shape of the scores is
        `(n_instances,)`. The higher the score, the more anomalous the instance. 'meta' contains information about
        the detector.
        """
        outputs = self.backend.predict(self.backend._to_tensor(X))
        output: Dict[str, Any] = {
            'data': outputs.numpy(),
            'meta': self.meta
        }
        return output
