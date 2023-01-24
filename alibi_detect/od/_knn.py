from typing import Callable, Union, Optional, Dict, Any, List, Tuple
from typing import TYPE_CHECKING

import numpy as np

from typing_extensions import Literal
from alibi_detect.base import outlier_prediction_dict
from alibi_detect.od.base import OutlierDetector, TransformProtocol, transform_protocols
from alibi_detect.od.pytorch import KNNTorch, Ensembler, to_numpy
from alibi_detect.od import normalizer_literals, aggregator_literals, get_aggregator, get_normalizer
from alibi_detect.utils.frameworks import BackendValidator
from alibi_detect.version import __version__


if TYPE_CHECKING:
    import torch


backends = {
    'pytorch': (KNNTorch, Ensembler)
}


class _KNN(OutlierDetector):
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

        The kNN detector is a non-parametric method for outlier detection. The detector computes the distance
        between each test point and its `k` nearest neighbors. The distance can be computed using a kernel function
        or a distance metric. The distance is then normalized and aggregated to obtain a single outlier score.

        The detector can be initialized with `k` a single value or an array of values. If `k` is a single value then
        the outlier score is the distance/kernel similarity to the k-th nearest neighbor. If `k` is an array of
        values then the outlier score is the distance/kernel similarity to each of the specified `k` neighbors.
        In the latter case, an aggregator must be specified to aggregate the scores.

        Parameters
        ----------
        k
            Number of neirest neighbors to compute distance to. `k` can be a single value or
            an array of integers. If an array is passed, an aggregator is required to aggregate
            the scores. If `k` is a single value the outlier score is the distance/kernel
            similarity to the `k`-th nearest neighbor. If `k` is a list then it returns the
            distance/kernel similarity to each of the specified `k` neighbors.
        kernel
            Kernel function to use for outlier detection. If ``None``, `torch.cdist` is used.
            Otherwise if a kernel is specified then instead of using `torch.cdist` the kernel
            defines the k nearest neighbor distance.
        normalizer
            Normalizer to use for outlier detection. If ``None``, no normalisation is applied.
            For a list of available normalizers, see :mod:`alibi_detect.od.pytorch.ensemble`.
        aggregator
            Aggregator to use for outlier detection. Can be set to ``None`` if `k` is a single
            value. For a list of available aggregators, see :mod:`alibi_detect.od.pytorch.ensemble`.
        backend
            Backend used for outlier detection. Defaults to ``'pytorch'``. Options are ``'pytorch'``.
        device
            Device type used. The default tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either ``'cuda'``, ``'gpu'`` or ``'cpu'``.

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

        backend_cls, ensembler_cls = backends[backend]
        ensembler = None

        if aggregator is None and isinstance(k, (list, np.ndarray, tuple)):
            raise ValueError('If `k` is a `np.ndarray`, `list` or `tuple`, '
                             'the `aggregator` argument cannot be ``None``.')

        if isinstance(k, (list, np.ndarray, tuple)):
            ensembler = ensembler_cls(
                normalizer=get_normalizer(normalizer),
                aggregator=get_aggregator(aggregator)
            )

        self.backend = backend_cls(k, kernel=kernel, ensembler=ensembler, device=device)

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

        Computes the k nearest neighbor distance/kernel similarity for each instance in `x`. If `k` is a single
        value then this is the score otherwise if `k` is an array of values then the score is aggregated using
        the ensembler.

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
        return to_numpy(score)

    def infer_threshold(self, x_ref: np.ndarray, fpr: float) -> None:
        """Infer the threshold for the kNN detector.

        The threshold is computed so that the outlier detector would incorectly classify `fpr` proportion of the
        reference data as outliers.

        Parameters
        ----------
        x_ref
            Reference data used to infer the threshold.
        fpr
            False positive rate used to infer the threshold. The false positive rate is the proportion of
            instances in `x_ref` that are incorrectly classified as outliers. The false positive rate should
            be in the range ``(0, 1)``.
        """
        self.backend.infer_threshold(self.backend._to_tensor(x_ref), fpr)

    def predict(self, x: np.ndarray) -> Dict[str, Any]:
        """Predict whether the instances in `x` are outliers or not.

        Scores the instances in `x` and if the threshold was inferred, returns the outlier labels and p-values as well.

        Parameters
        ----------
        x
            Data to predict. The shape of `x` should be `(n_instances, n_features)`.

        Returns
        -------
        Dictionary with keys 'data' and 'meta'. 'data' contains the outlier scores. If threshold inference was  \
        performed, 'data' also contains the threshold value, outlier labels and p-vals . The shape of the scores is \
        `(n_instances,)`. The higher the score, the more anomalous the instance. 'meta' contains information about \
        the detector.
        """
        outputs = self.backend.predict(self.backend._to_tensor(x))
        output = outlier_prediction_dict()
        output['data'] = {
            **output['data'],
            **to_numpy(outputs)
        }
        output['meta'] = {
            **output['meta'],
            'name': self.__class__.__name__,
            'detector_type': 'outlier',
            'online': False,
            'version': __version__,
        }
        return output
