import numpy as np
from alibi_detect.od.sklearn.base import SklearnOutlierDetector
from sklearn.mixture import GaussianMixture


class GMMSklearn(SklearnOutlierDetector):
    def __init__(
            self,
            n_components: int,
            ):
        """sklearn backend for GMM detector.

        Parameters
        ----------
        n_components
            Number of components in guassian mixture model.
        """
        SklearnOutlierDetector.__init__(self)
        self.n_components = n_components
        self.gmm = None

    def _fit(self, x_ref: np.ndarray) -> None:
        """Fit the outlier detector to the reference data.

        Parameters
        ----------
        x_ref
            Reference data.
        """
        self.gmm = GaussianMixture(n_components=self.n_components).fit(x_ref)

    def score(self, x: np.ndarray) -> np.ndarray:
        """Score the data.

        Parameters
        ----------
        x
            Data to score.
        """
        self.check_fitted()
        return - self.gmm.score_samples(x)
