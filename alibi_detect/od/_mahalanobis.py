from typing import Union, Optional, Dict, Any
from typing import TYPE_CHECKING

import numpy as np

from alibi_detect.utils._types import Literal
from alibi_detect.base import BaseDetector, FitMixin, ThresholdMixin, outlier_prediction_dict
from alibi_detect.od.pytorch import MahalanobisTorch
from alibi_detect.utils.frameworks import BackendValidator
from alibi_detect.version import __version__


if TYPE_CHECKING:
    import torch


backends = {
    'pytorch': MahalanobisTorch
}


class Mahalanobis(BaseDetector, FitMixin, ThresholdMixin,):
    def __init__(
        self,
        min_eigenvalue: float = 1e-6,
        device: Optional[Union[Literal['cuda', 'gpu', 'cpu'], 'torch.device']] = None,
        backend: Literal['pytorch'] = 'pytorch',
    ) -> None:
        """The Mahalanobis outlier detection method.

        The Mahalanobis method computes the covariance matrix of a reference dataset passed in the `fit` method. It
        then saves the eigenvectors of this matrix with eigenvalues greater than `min_eigenvalue`. While doing so
        it also scales the eigenvectors such that the reference data projected onto them has mean ``0`` and std ``1``.

        When we score a test point `x` we project it onto the eigenvectors and compute the l2-norm of the
        projected point. The higher the score, the more outlying the instance.

        Parameters
        ----------
        min_eigenvalue
            Eigenvectors with eigenvalues below this value will be discarded.
        backend
            Backend used for outlier detection. Defaults to ``'pytorch'``. Options are ``'pytorch'``.
        device
            Device type used. The default tries to use the GPU and falls back on CPU if needed. Can be specified by
            passing either ``'cuda'``, ``'gpu'`` or ``'cpu'``.

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
        self.backend = backend_cls(
            min_eigenvalue,
            device=device
        )

    def fit(self, x_ref: np.ndarray) -> None:
        """Fit the detector on reference data.

        Fitting the Mahalanobis method amounts to computing the covariance matrix of the reference data and
        saving the eigenvectors with eigenvalues greater than `min_eigenvalue`.

        Parameters
        ----------
        x_ref
            Reference data used to fit the detector.
        """
        self.backend.fit(self.backend._to_tensor(x_ref))

    def score(self, x: np.ndarray) -> np.ndarray:
        """Score `x` instances using the detector.

        The mahalanobis method projects `x` onto the eigenvectors of the covariance matrix of the reference data.
        The score is then the l2-norm of the projected data. The higher the score, the more outlying the instance.

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
        """Infer the threshold for the Mahalanobis detector.

        The threshold is inferred using the reference data and the false positive rate. The threshold is used to
        determine the outlier labels in the predict method.

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
