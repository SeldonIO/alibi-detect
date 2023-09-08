from typing import Dict, Any
from alibi_detect.exceptions import _catch_error as catch_error
from typing_extensions import Literal

import numpy as np

from alibi_detect.base import BaseDetector, FitMixin, ThresholdMixin, outlier_prediction_dict
from alibi_detect.od.pytorch import MahalanobisTorch
from alibi_detect.utils.frameworks import BackendValidator
from alibi_detect.version import __version__
from alibi_detect.utils._types import TorchDeviceType


backends = {
    'pytorch': MahalanobisTorch
}


class Mahalanobis(BaseDetector, FitMixin, ThresholdMixin):
    def __init__(
        self,
        min_eigenvalue: float = 1e-6,
        backend: Literal['pytorch'] = 'pytorch',
        device: TorchDeviceType = None,
    ) -> None:
        """
        The Mahalanobis outlier detection method.

        The Mahalanobis detector computes the directions of variation of a dataset and uses them to detect when points
        are outliers by checking to see if the points vary from dataset points in unexpected ways.

        When we fit the Mahalanobis detector we compute the covariance matrix of the reference data and its eigenvectors
        and eigenvalues. We filter small eigenvalues for numerical stability using the `min_eigenvalue` parameter. We
        then inversely weight each eigenvector by its eigenvalue.

        When we score test points we project them onto the eigenvectors and compute the l2-norm of the projected point.
        Because the eigenvectors are inversely weighted by the eigenvalues, the score will take into account the
        difference in variance along each direction of variation. If a test point lies along a direction of high
        variation then it must lie very far out to obtain a high score. If a test point lies along a direction of low
        variation then it doesn't need to lie very far out to obtain a high score.

        Parameters
        ----------
        min_eigenvalue
            Eigenvectors with eigenvalues below this value will be discarded. This is to ensure numerical stability.
        backend
            Backend used for outlier detection. Defaults to ``'pytorch'``. Options are ``'pytorch'``.
        device
            Device type used. The default tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of
            ``torch.device``.

        Raises
        ------
        NotImplementedError
            If choice of `backend` is not implemented.
        """
        super().__init__()

        backend_str: str = backend.lower()
        BackendValidator(
            backend_options={'pytorch': ['pytorch']},
            construct_name=self.__class__.__name__
        ).verify_backend(backend_str)

        backend_cls = backends[backend]
        self.backend = backend_cls(min_eigenvalue, device=device)

        # set metadata
        self.meta['detector_type'] = 'outlier'
        self.meta['data_type'] = 'numeric'
        self.meta['online'] = False

    def fit(self, x_ref: np.ndarray) -> None:
        """Fit the detector on reference data.

        Fitting the Mahalanobis detector amounts to computing the covariance matrix and its eigenvectors. We filter out
        very small eigenvalues using the `min_eigenvalue` parameter. We then scale the eigenvectors such that the data
        projected onto them has mean ``0`` and std ``1``.

        Parameters
        ----------
        x_ref
            Reference data used to fit the detector.
        """
        self.backend.fit(self.backend._to_backend_dtype(x_ref))

    @catch_error('NotFittedError')
    def score(self, x: np.ndarray) -> np.ndarray:
        """Score `x` instances using the detector.

        The mahalanobis method projects `x` onto the scaled eigenvectors computed during the fit step. The score is then
        the l2-norm of the projected data. The higher the score, the more outlying the instance.

        Parameters
        ----------
        x
            Data to score. The shape of `x` should be `(n_instances, n_features)`.

        Returns
        -------
        Outlier scores. The shape of the scores is `(n_instances,)`. The higher the score, the more outlying the \
        instance.

        Raises
        ------
        NotFittedError
            If called before detector has been fit.
        """
        score = self.backend.score(self.backend._to_backend_dtype(x))
        return self.backend._to_frontend_dtype(score)

    @catch_error('NotFittedError')
    def infer_threshold(self, x: np.ndarray, fpr: float) -> None:
        """Infer the threshold for the Mahalanobis detector.

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
