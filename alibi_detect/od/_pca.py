from typing import Union, Optional, Callable, Dict, Any
from typing import TYPE_CHECKING

import numpy as np

from alibi_detect.utils._types import Literal
from alibi_detect.base import outlier_prediction_dict
from alibi_detect.od.base import OutlierDetector
from alibi_detect.od.pytorch.base import to_numpy
from alibi_detect.od.pytorch import KernelPCATorch, LinearPCATorch
from alibi_detect.utils.frameworks import BackendValidator
from alibi_detect.version import __version__


if TYPE_CHECKING:
    import torch


backends = {
    'pytorch': (KernelPCATorch, LinearPCATorch)
}


class _PCA(OutlierDetector):
    def __init__(
        self,
        n_components: int,
        kernel: Optional[Callable] = None,
        device: Optional[Union[Literal['cuda', 'gpu', 'cpu'], 'torch.device']] = None,
        backend: Literal['pytorch'] = 'pytorch',
    ) -> None:
        """Principal Component Analysis (PCA) outlier detector.

        The detector is based on the Principal Component Analysis (PCA) algorithm. There are two variants of PCA:
        linear PCA and kernel PCA. Linear PCA computes the eigenvectors of the covariance matrix of the data. Kernel
        PCA computes the eigenvectors of the kernel matrix of the data. In each case, we choose the smallest
        `n_components` eigenvectors. We do this as they correspond to the invariant directions of the data. i.e the
        directions along which the data is least spread out. Thus a point that deviates along these dimensions is more
        likely to be an outlier.

        When scoring a test instance we project it onto the eigenvectors and compute its score using the L2 norm. If
        a threshold is fitted we use this to determine whether the instance is an outlier or not.

        Parameters
        ----------
        n_components:
            The number of dimensions in the principle subspace. For linear pca should have
            ``1 <= n_components < dim(data)``. For kernel pca should have ``1 <= n_components < len(data)``.
        backend
            Backend used for outlier detection. Defaults to ``'pytorch'``. Options are ``'pytorch'``.
        kernel
            Kernel function to use for outlier detection. If ``None``, linear PCA is used instead of the
            kernel variant.
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

        kernel_backend_cls, linear_backend_cls = backends[backend]

        self.backend: Union[KernelPCATorch, LinearPCATorch]
        if kernel is not None:
            self.backend = kernel_backend_cls(
                n_components=n_components,
                device=device,
                kernel=kernel
            )
        else:
            self.backend = linear_backend_cls(
                n_components=n_components,
                device=device,
            )

    def fit(self, x_ref: np.ndarray) -> None:
        """Fit the detector on reference data.

        Compute the eigenvectors of the covariance/kernel matrix of `x_ref` and save the smallest `n_components`
        eigenvectors.

        Parameters
        ----------
        x_ref
            Reference data used to fit the detector.
        """
        self.backend.fit(self.backend._to_tensor(x_ref))

    def score(self, x: np.ndarray) -> np.ndarray:
        """Score `x` instances using the detector.

        Project `x` onto the eigenvectors and compute its score using the L2 norm.

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
        """Infer the threshold for the Mahalanobis detector.

        The threshold is set such that the false positive rate of the detector on the reference data is `fpr`.


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
