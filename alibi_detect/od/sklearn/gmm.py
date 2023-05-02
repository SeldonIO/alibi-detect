import numpy as np
from typing import Dict
from alibi_detect.od.sklearn.base import SklearnOutlierDetector
from sklearn.mixture import GaussianMixture


class GMMSklearn(SklearnOutlierDetector):
    def __init__(
        self,
        n_components: int,
    ):
        """sklearn backend for the Gaussian Mixture Model (GMM) outlier detector.

        Parameters
        ----------
        n_components
            Number of components in guassian mixture model.
        """
        super().__init__()
        self.n_components = n_components

    def _fit(
        self,
        x_ref: np.ndarray,
        tol: float = 1e-3,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: str = 'kmeans',
        verbose: int = 0,
    ) -> None:
        """Fit the outlier detector to the reference data.

        Parameters
        ----------
        x_ref
            Reference data.
        tol
            Convergence threshold. EM iterations will stop when the lower bound average gain is below this threshold.
        max_iter
            Maximum number of EM iterations to perform.
        n_init
            Number of initializations to perform.
        init_params
            Method used to initialize the weights, the means and the precisions. Must be one of:
            'kmeans' : responsibilities are initialized using kmeans.
            'kmeans++' : responsibilities are initialized using kmeans++.
            'random' : responsibilities are initialized randomly.
            'random_from_data' : responsibilities are initialized randomly from the data.
        verbose
            Enable verbose output. If 1 then it prints the current initialization and each iteration step. If greater
            than 1 then it prints also the log probability and the time needed for each step.

        """
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            tol=tol,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            verbose=verbose,
        )
        self.gmm = self.gmm.fit(
            x_ref,
        )

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
            tol=fit_kwargs.get('tol', 1e-3),
            max_iter=(lambda v: 100 if v is None else v)(fit_kwargs.get('epochs', None)),
            n_init=fit_kwargs.get('n_init', 1),
            init_params=fit_kwargs.get('init_params', 'kmeans'),
            verbose=fit_kwargs.get('verbose', 0),
        )

    def score(self, x: np.ndarray) -> np.ndarray:
        """Score the data.

        Parameters
        ----------
        x
            Data to score.
        """
        self.check_fitted()
        return - self.gmm.score_samples(x)
