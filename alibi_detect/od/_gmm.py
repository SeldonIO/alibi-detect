from typing import Union, Optional, Dict, Any, TYPE_CHECKING

import numpy as np

from alibi_detect.utils._types import Literal
from alibi_detect.base import outlier_prediction_dict
from alibi_detect.base import BaseDetector, ThresholdMixin, FitMixin
from alibi_detect.od.pytorch import GMMTorch
from alibi_detect.od.sklearn import GMMSklearn
from alibi_detect.utils.frameworks import BackendValidator
from alibi_detect.version import __version__


if TYPE_CHECKING:
    import torch


backends = {
    'pytorch': GMMTorch,
    'sklearn': GMMSklearn
}


class GMM(BaseDetector, ThresholdMixin, FitMixin):
    def __init__(
        self,
        n_components: int = 1,
        device: Optional[Union[Literal['cuda', 'gpu', 'cpu'], 'torch.device']] = None,
        backend: Literal['pytorch', 'sklearn'] = 'pytorch',
    ) -> None:
        """Gaussian Mixture Model (GMM) outlier detector.

        The guassian mixture model outlier detector fits a mixture of gaussian distributions to the reference data.
        Test points are scored via the negative log-likhood under the corresponding density function.

        We support two backends: ``'pytorch'`` and ``'sklearn'``. The ``'pytorch'`` backend allows for GPU acceleration
        and uses gradient descent to fit the mixture of gaussians. We recommend using the ``'pytorch'`` backend for
        for large datasets. The ``'sklearn'`` backend is a pure python implementation and is recommended for smaller
        datasets.

        Parameters
        ----------
        n_components:
            The number of mixture components. Defaults to ``1``.
        backend
            Backend used for outlier detection. Defaults to ``'pytorch'``. Options are ``'pytorch'`` and ``'sklearn'``.
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
            backend_options={'pytorch': ['pytorch'], 'sklearn': ['sklearn']},
            construct_name=self.__class__.__name__
        ).verify_backend(backend_str)

        backend_cls = backends[backend]
        args: Dict[str, Any] = {'n_components': n_components}
        if backend == 'pytorch':
            args['device'] = device
        self.backend = backend_cls(**args)

    def fit(
        self,
        x_ref: np.ndarray,
        optimizer: Optional[str] = 'Adam',
        learning_rate: float = 0.1,
        batch_size: Optional[int] = None,
        epochs: Optional[int] = None,
        tol: float = 1e-3,
        n_init: int = 1,
        init_params: str = 'kmeans',
        verbose: int = 0,
    ) -> None:
        """Fit the detector on reference data.

        If the ``'pytorch'`` backend is used, the detector is fitted using gradient descent. This is the recommended
        backend for larger datasets.

        If the ``'sklearn'`` backend is used, the detector is fitted using the EM algorithm. The ``'sklearn'``
        backend is recommended for smaller datasets. For more information on the EM algorithm and the sklearn Gaussian
        Mixture Model, see `here <https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture>`_.  # noqa: E501

        Parameters
        ----------
        x_ref
            Reference data used to fit the detector.
        optimizer
            Optimizer used to fit the detector. Only used if the ``'pytorch'`` backend is used. Defaults to ``'Adam'``.
        learning_rate
            Learning rate used to fit the detector. Only used if the ``'pytorch'`` backend is used. Defaults to ``0.1``.
        batch_size
            Batch size used to fit the detector. Only used if the ``'pytorch'`` backend is used. Defaults to ``None``.
            If ``None``, the entire dataset is used in each epoch.
        epochs
            Number of epochs used to fit the detector. Used for both the ``'pytorch'`` and ``'sklearn'`` backends.
            If the backend is ``'sklearn'``, the detector is fitted using the EM algorithm and the number of epochs
            defualts to ``10``. If the backend is ``'pytorch'``, the detector is fitted using gradient descent and
            the number of epochs defaults to ``100``.
        tol
            Tolerance used to fit the detector. Only used if the ``'sklearn'`` backend is used. Defaults to ``1e-3``.
        n_init
            Number of initializations used to fit the detector. Only used if the ``'sklearn'`` backend is used.
            Defaults to ``1``.
        init_params
            Initialization method used to fit the detector. Only used if the ``'sklearn'`` backend is used. Must be
            one of:
            'kmeans' : responsibilities are initialized using kmeans.
            'kmeans++' : responsibilities are initialized using kmeans++.
            'random' : responsibilities are initialized randomly.
            'random_from_data' : responsibilities are initialized randomly from the data.
            Defaults to ``'kmeans'``.
        verbose
            Verbosity level used to fit the detector. Only used if the ``'sklearn'`` backend is used. Defaults to ``0``.
        """
        self.backend.fit(
            self.backend._to_tensor(x_ref),
            **self.backend.format_fit_kwargs(locals())
        )

    def score(self, x: np.ndarray) -> np.ndarray:
        """Score `x` instances using the detector.

        To score an instance, we compute the negative log-likelihood under the corresponding density function of
        the fitted gaussian mixture model.

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
        """Infer the threshold for the GMM detector.

        The threshold is computed so that the outlier detector would incorectly classify `fpr` proportion of the
        reference data as outliers.

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

        Scores the instances in `x` and if the threshold was inferred, returns the outlier labels and p-values as well.

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
