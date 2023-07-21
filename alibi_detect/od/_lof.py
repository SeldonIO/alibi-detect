from typing import Callable, Union, Optional, Dict, Any, List, Tuple
from typing_extensions import Literal

import numpy as np

from alibi_detect.base import outlier_prediction_dict
from alibi_detect.exceptions import _catch_error as catch_error
from alibi_detect.od.base import TransformProtocol, TransformProtocolType
from alibi_detect.base import BaseDetector, FitMixin, ThresholdMixin
from alibi_detect.od.pytorch import LOFTorch, Ensembler
from alibi_detect.od.base import get_aggregator, get_normalizer, NormalizerLiterals, AggregatorLiterals
from alibi_detect.utils.frameworks import BackendValidator
from alibi_detect.version import __version__
from alibi_detect.utils._types import TorchDeviceType


backends = {
    'pytorch': (LOFTorch, Ensembler)
}


class LOF(BaseDetector, FitMixin, ThresholdMixin):
    def __init__(
        self,
        k: Union[int, np.ndarray, List[int], Tuple[int]],
        kernel: Optional[Callable] = None,
        normalizer: Optional[Union[TransformProtocolType, NormalizerLiterals]] = 'PValNormalizer',
        aggregator: Union[TransformProtocol, AggregatorLiterals] = 'AverageAggregator',
        backend: Literal['pytorch'] = 'pytorch',
        device: TorchDeviceType = None,
    ) -> None:
        """
        Local Outlier Factor (LOF) outlier detector.

        The LOF detector is a non-parametric method for outlier detection. It computes the local density
        deviation of a given data point with respect to its neighbors. It considers as outliers the
        samples that have a substantially lower density than their neighbors.

        The detector can be initialized with `k` a single value or an array of values. If `k` is a single value then
        the score method uses the distance/kernel similarity to the k-th nearest neighbor. If `k` is an array of
        values then the score method uses the distance/kernel similarity to each of the specified `k` neighbors.
        In the latter case, an `aggregator` must be specified to aggregate the scores.

        Note that, in the multiple k case, a normalizer can be provided. If a normalizer is passed then it is fit in
        the `infer_threshold` method and so this method must be called before the `predict` method. If this is not
        done an exception is raised. If `k` is a single value then the predict method can be called without first
        calling `infer_threshold` but only scores will be returned and not outlier predictions.

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
            Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of
            ``torch.device``.

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
        self.backend.fit(self.backend._to_backend_dtype(x_ref))

    @catch_error('NotFittedError')
    @catch_error('ThresholdNotInferredError')
    def score(self, x: np.ndarray) -> np.ndarray:
        """Score `x` instances using the detector.

        Computes the local outlier factor for each point in `x`. This is the density of each point `x`
        relative to those of its neighbors in `x_ref`. If `k` is an array of values then the score for
        each `k` is aggregated using the ensembler.

        Parameters
        ----------
        x
            Data to score. The shape of `x` should be `(n_instances, n_features)`.

        Returns
        -------
        Outlier scores. The shape of the scores is `(n_instances,)`. The higher the score, the more anomalous the \
        instance.

        Raises
        ------
        NotFittedError
            If called before detector has been fit.
        ThresholdNotInferredError
            If k is a list and a threshold was not inferred.
        """
        score = self.backend.score(self.backend._to_backend_dtype(x))
        score = self.backend._ensembler(score)
        return self.backend._to_frontend_dtype(score)

    @catch_error('NotFittedError')
    def infer_threshold(self, x: np.ndarray, fpr: float) -> None:
        """Infer the threshold for the LOF detector.

        The threshold is computed so that the outlier detector would incorrectly classify `fpr` proportion of the
        reference data as outliers.

        Parameters
        ----------
        x
            Reference data used to infer the threshold.
        fpr
            False positive rate used to infer the threshold. The false positive rate is the proportion of
            instances in `x` that are incorrectly classified as outliers. The false positive rate should
            be in the range ``(0, 1)``.

        Raises
        ------
        ValueError
            Raised if `fpr` is not in ``(0, 1)``.
        NotFittedError
            If called before detector has been fit.
        """
        self.backend.infer_threshold(self.backend._to_backend_dtype(x), fpr)

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

        Raises
        ------
        NotFittedError
            If called before detector has been fit.
        ThresholdNotInferredError
            If k is a list and a threshold was not inferred.
        """
        outputs = self.backend.predict(self.backend._to_backend_dtype(x))
        output = outlier_prediction_dict()
        output['data'] = {
            **output['data'],
            **self.backend._to_frontend_dtype(outputs)
        }
        output['meta'] = {
            **output['meta'],
            'name': self.__class__.__name__,
            'detector_type': 'outlier',
            'online': False,
            'version': __version__,
        }
        return output
