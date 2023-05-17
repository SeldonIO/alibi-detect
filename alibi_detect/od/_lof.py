from typing import Callable, Union, Optional, Dict, Any, List, Tuple
from typing import TYPE_CHECKING

import numpy as np

from typing_extensions import Literal
from alibi_detect.exceptions import _catch_error as catch_error
from alibi_detect.base import outlier_prediction_dict
from alibi_detect.od.base import TransformProtocol, TransformProtocolType
from alibi_detect.base import BaseDetector, FitMixin, ThresholdMixin
from alibi_detect.od.pytorch import LOFTorch, Ensembler
from alibi_detect.od.base import get_aggregator, get_normalizer, NormalizerLiterals, AggregatorLiterals
from alibi_detect.utils.frameworks import BackendValidator
from alibi_detect.version import __version__


if TYPE_CHECKING:
    import torch


backends = {
    'pytorch': (LOFTorch, Ensembler)
}


class LOF(BaseDetector, FitMixin, ThresholdMixin):
    def __init__(
        self,
        k: Union[int, np.ndarray, List[int], Tuple[int]],
        kernel: Optional[Callable] = None,
        normalizer: Optional[Union[TransformProtocolType, NormalizerLiterals]] = 'ShiftAndScaleNormalizer',
        aggregator: Union[TransformProtocol, AggregatorLiterals] = 'AverageAggregator',
        device: Optional[Union[Literal['cuda', 'gpu', 'cpu'], 'torch.device']] = None,
        backend: Literal['pytorch'] = 'pytorch',
    ) -> None:
        """
        Local Outlier Factor (LOF) outlier detector.

        The LOF detector is a non-parametric method for outlier detection. It computes the local density
        deviation of a given data point with respect to its neighbors. It considers as outliers the
        samples that have a substantially lower density than their neighbors.

        Parameters
        ----------
        k
            Number of nearest neighbors to compute distance to. `k` can be a single value or
            an array of integers. If an array is passed, an aggregator is required to aggregate
            the scores. If `k` is a single value we compute the local outlier factor for that `k`.
            Otherwise if `k` is a list then we compute and aggregate the local outlier factor for each
            value in `k`.
        kernel
            Kernel function to use for outlier detection. If ``None``, `torch.cdist` is used.
            Otherwise if a kernel is specified then instead of using `torch.cdist` the kernel
            defines the k nearest neighbor distance.
        normalizer
            Normalizer to use for outlier detection. If ``None``, no normalization is applied.
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

        # set metadata
        self.meta['detector_type'] = 'outlier'
        self.meta['data_type'] = 'numeric'
        self.meta['online'] = False

    def fit(self, x_ref: np.ndarray) -> None:
        """Fit the detector on reference data.

        Parameters
        ----------
        x_ref
            Reference data used to fit the detector.
        """
        self.backend.fit(self.backend._to_tensor(x_ref))

    @catch_error('NotFittedError')
    @catch_error('ThresholdNotInferredError')
    def score(self, X: np.ndarray) -> np.ndarray:
        """Score `x` instances using the detector.

        The LOF detector scores the instances in `x` by computing the local outlier factor for each instance. The
        higher the score, the more anomalous the instance.

        Parameters
        ----------
        x
            Data to score. The shape of `x` should be `(n_instances, n_features)`.

        Returns
        -------
        Outlier scores. The shape of the scores is `(n_instances,)`. The higher the score, the more anomalous the \
        instance.
        """
        score = self.backend.score(self.backend._to_tensor(X))
        score = self.backend._ensembler(score)
        return self.backend._to_numpy(score)

    @catch_error('NotFittedError')
    def infer_threshold(self, X: np.ndarray, fpr: float) -> None:
        """Infer the threshold for the LOF detector.

        The threshold is computed so that the outlier detector would incorrectly classify `fpr` proportion of the
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
        self.backend.infer_threshold(self.backend._to_tensor(X), fpr)

    @catch_error('NotFittedError')
    @catch_error('ThresholdNotInferredError')
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
