from typing import Callable, Union, Optional, Dict, Any, List, Tuple
from typing import TYPE_CHECKING

import numpy as np

from typing_extensions import Literal
from alibi_detect.base import outlier_prediction_dict
from alibi_detect.od.base import OutlierDetector, TransformProtocol, transform_protocols
from alibi_detect.od.pytorch import KNNTorch, Accumulator
from alibi_detect.od import normalizer_literals, aggregator_literals, get_aggregator, get_normalizer
from alibi_detect.utils.frameworks import BackendValidator
from alibi_detect.version import __version__


if TYPE_CHECKING:
    import torch


backends = {
    'pytorch': (KNNTorch, Accumulator)
}


class KNN(OutlierDetector):
    def __init__(
        self,
        k: Union[int, np.ndarray, List[int], Tuple[int]],
        kernel: Optional[Callable] = None,
        normalizer: Optional[Union[transform_protocols, normalizer_literals]] = 'ShiftAndScaleNormalizer',
        aggregator: Union[TransformProtocol, aggregator_literals] = 'AverageAggregator',
        device: Optional[Union[Literal['cuda', 'gpu', 'cpu'], 'torch.device']] = None,
        backend: Literal['pytorch'] = 'pytorch',
    ) -> None:
        """
        k-Nearest Neighbours (kNN) outlier detector.

        Parameters
        ----------
        k
            Number of nearest neighbours to use for outlier detection. If an array is passed, an
            aggregator is required to aggregate the scores.
        kernel
            Kernel function to use for outlier detection. If ``None``, `torch.cdist` is used.
        normalizer
            Normalizer to use for outlier detection. If ``None``, no normalisation is applied.
        aggregator
            Aggregator to use for outlier detection. Can be set to ``None`` if `k` is a single value.
        backend
            Backend used for outlier detection. Defaults to ``'pytorch'``. Options are ``'pytorch'``.
        device
            Device type used. The default tries to use the GPU and falls back on CPU if needed. Can be specified by
            passing either ``'cuda'``, ``'gpu'`` or ``'cpu'``.

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

        if aggregator is None and isinstance(k, (list, np.ndarray, tuple)):
            raise ValueError('If `k` is a `np.ndarray`, `list` or `tuple`, '
                             'the `aggregator` argument cannot be ``None``.')

        if isinstance(k, (list, np.ndarray, tuple)):
            accumulator = accumulator_cls(
                normalizer=get_normalizer(normalizer),
                aggregator=get_aggregator(aggregator)
            )

        self.backend = backend_cls(k, kernel=kernel, accumulator=accumulator, device=device)

    def fit(self, x_ref: np.ndarray) -> None:
        """Fit the detector on reference data.

        Parameters
        ----------
        x_ref
            Reference data used to fit the detector.
        """
        self.backend.fit(self.backend._to_tensor(x_ref))

    def score(self, x: np.ndarray) -> np.ndarray:
        """Score `x` instances using the detector.

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

    def infer_threshold(self, x_ref: np.ndarray, fpr: float) -> None:
        """Infer the threshold for the kNN detector. The threshold is inferred using the reference data and the false
        positive rate. The threshold is used to determine the outlier labels in the predict method.

        Parameters
        ----------
        x_ref
            Reference data used to infer the threshold.
        fpr
            False positive rate used to infer the threshold. The false positive rate is the proportion of instances in \
            `x_ref` that are incorrectly classified as outliers. The false positive rate should be in the range \
            ``(0, 1)``.
        """
        self.backend.infer_threshold(self.backend._to_tensor(x_ref), fpr)

    def predict(self, x: np.ndarray) -> Dict[str, Any]:
        """Predict whether the instances in `x` are outliers or not.

        Parameters
        ----------
        x
            Data to predict. The shape of `x` should be `(n_instances, n_features)`.

        Returns
        -------
        Dictionary with keys 'data' and 'meta'. 'data' contains the outlier scores. If threshold inference was  \
        performed, 'data' also contains the threshold value, outlier labels and p_vals . The shape of the scores is \
        `(n_instances,)`. The higher the score, the more anomalous the instance. 'meta' contains information about \
        the detector.
        """
        outputs = self.backend.predict(self.backend._to_tensor(x))
        output = outlier_prediction_dict()
        output['data'] = {
            **output['data'],
            **self.backend._to_numpy(outputs)
        }
        output['meta'] = {
            **output['meta'],
            'name': self.__class__.__name__,
            'detector_type': 'outlier',
            'online': False,
            'version': __version__,
        }
        return output
