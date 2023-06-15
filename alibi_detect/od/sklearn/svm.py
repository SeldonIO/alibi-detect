import numpy as np
from typing import Dict, Union, Callable, Optional
from typing_extensions import Literal
from sklearn.linear_model import SGDOneClassSVM

from alibi_detect.od.sklearn.base import SklearnOutlierDetector
from sklearn.kernel_approximation import Nystroem


class SVMSklearn(SklearnOutlierDetector):
    ensemble = False

    def __init__(
        self,
        n_components: Optional[int] = None,
        kernel: Union[Literal['linear', 'poly', 'rbf', 'sigmoid'], Callable] = 'rbf',
        sigma: Optional[float] = None,
        kernel_params: Optional[Dict] = None,
    ):
        """SKlearn backend for the One class support vector machine (SVM) outlier detector.

        Parameters
        ----------
        n_components
            Number of features to construct. How many data points will be used to construct the mapping.
        kernel
            Kernel map to be approximated by the Nystroem method. A callable should accept two arguments and the
            keyword arguments passed to this object as ``kernel_params``, and should return a floating point number.
            Otherwise the user can pass a string and the corresponding built-in kernel map will be used from
            `sklearn.metrics.pairwise <https://scikit-learn.org/stable/modules/classes.html#pairwise-metrics/>`_.
            See also
            `sklearn.kernel_approximation.Nystroem <https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html/>`_  # noqa
        sigma
            Sigma parameter for the RBF, polynomial, and sigmoid kernels. Interpretation of the default value is left
            to the kernel; see the documentation for
            `sklearn.metrics.pairwise <https://scikit-learn.org/stable/modules/classes.html#pairwise-metrics/>`_.
            Ignored by other kernels.
        kernel_params
            Additional parameters (keyword arguments) for kernel function passed as callable object.

        Raises
        ------
        ValueError
            If `n_components` is less than 1.
        """
        super().__init__()
        self.n_components = n_components
        self.kernel = kernel
        self.sigma = sigma
        self.nystroem = None
        self.kernel_params = kernel_params

    def fit(  # type: ignore[override]
        self,
        x_ref: np.ndarray,
        nu: float = 0.5,
        tol: float = 1e-6,
        max_iter: int = 1000,
        verbose: int = 0,
    ) -> Dict:
        """Fit the SKLearn SVM model.

        Parameters
        ----------
        x_ref
            Training data.
        nu
            The proportion of the training data that should be considered outliers. Note that this does
            not necessarily correspond to the false positive rate on test data, which is still defined when
            calling the `infer_threshold` method.
        tol
            The decrease in loss required over the previous ``n_iter_no_change`` iterations in order to
            continue optimizing.
        max_iter
            The maximum number of optimization steps.
        verbose
            Verbosity level during training. ``0`` is silent, ``1`` a progress bar.

        Returns
        -------
        Dictionary with fit results. The dictionary contains the following keys:
            - converged: `bool` indicating whether training converged.
            - n_iter: number of iterations performed.
        """
        n_components = self.n_components if self.n_components is not None else len(x_ref)
        self.nystroem = Nystroem(
            kernel=self.kernel,
            n_components=n_components,
            gamma=1. / (2. * self.sigma ** 2) if self.sigma is not None else None,
            kernel_params=self.kernel_params,
        )
        x_ref = self.nystroem.fit(x_ref).transform(x_ref)
        self.gmm = SGDOneClassSVM(
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
            nu=nu
        )
        self.gmm = self.gmm.fit(x_ref)
        self._set_fitted()
        return {
            'converged': self.gmm.n_iter_ < max_iter,
            'n_iter': self.gmm.n_iter_,
        }

    def format_fit_kwargs(self, fit_kwargs: Dict) -> Dict:
        """Format kwargs for `fit` method.

        Parameters
        ----------
        kwargs
            dictionary of Kwargs to format. See `fit` method for details.

        Returns
        -------
        Formatted kwargs.
        """
        return dict(
            nu=fit_kwargs.get('nu', 0.5),
            tol=fit_kwargs.get('tol', 1e-3),
            max_iter=fit_kwargs.get('max_iter', 1000),
            verbose=fit_kwargs.get('verbose', 0),
        )

    def score(self, x: np.ndarray) -> np.ndarray:
        """Computes the score of `x`

        Parameters
        ----------
        x
            `np.ndarray` with leading batch dimension.

        Returns
        -------
        `np.ndarray` of scores with leading batch dimension.

        Raises
        ------
        NotFittedError
            Raised if method called and detector has not been fit.
        """
        self.check_fitted()
        x = self.nystroem.transform(x)
        return - self.gmm.score_samples(x)
