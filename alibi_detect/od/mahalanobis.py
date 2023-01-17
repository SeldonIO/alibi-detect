from typing import Union, Optional, Dict, Any
from typing import TYPE_CHECKING

import numpy as np

from alibi_detect.utils._types import Literal
from alibi_detect.base import outlier_prediction_dict
from alibi_detect.od.base import OutlierDetector
from alibi_detect.od.pytorch.base import to_numpy
from alibi_detect.od.pytorch.mahalanobis import MahalanobisTorch
from alibi_detect.utils.frameworks import BackendValidator
from alibi_detect.version import __version__


if TYPE_CHECKING:
    import torch


backends = {
    'pytorch': MahalanobisTorch
}


class Mahalanobis(OutlierDetector):
    def __init__(
        self,
        min_eigenvalue: float = 1e-6,
        device: Optional[Union[Literal['cuda', 'gpu', 'cpu'], 'torch.device']] = None,
        backend: Literal['pytorch'] = 'pytorch',
    ) -> None:
        """
        Outliers identified via Mahalanobis distance.

        The linear variant can be interpreted as projecting onto (orthogonal) eigenvectors of the
        covariance matrix, scaled such that projections onto the eigenvectors have mean 0 and std 1.
        The Mahalanobis distance is then the l2-norm from the origin.

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
        return to_numpy(score)

    def infer_threshold(self, x_ref: np.ndarray, fpr: float) -> None:
        """Infer the threshold for the Mahalanobis detector. The threshold is inferred using the reference data
        and the false positive rate. The threshold is used to determine the outlier labels in the predict method.

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
