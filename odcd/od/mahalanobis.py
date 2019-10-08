import logging
import numpy as np
from scipy.linalg import eigh

logger = logging.getLogger(__name__)

EPSILON = 1e-8

# TODO: include fit step for categorical variable distance. This would allow to apply the detector to num + cat data.


class Mahalanobis:

    def __init__(self,
                 threshold,
                 n_components: int = 3,
                 std_clip: int = 3,
                 start_clip: int = 100,
                 max_n: int = None) -> None:
        """
        Outlier detector for tabular data using the Mahalanobis distance.

        Parameters
        ----------
        threshold
            Mahalanobis distance threshold used to classify outliers.
        n_components
            Number of principal components used.
        std_clip
            Feature-wise stdev used to clip the observations before updating the mean and cov.
        start_clip
            Number of observations before clipping is applied.
        max_n
            Algorithm behaves as if it has seen at most max_n points.
        """

        self.threshold = threshold
        self.n_components = n_components
        self.std_clip = std_clip
        self.start_clip = start_clip
        self.max_n = max_n

        # initial parameter values
        self.clip = None
        self.mean = 0
        self.C = 0
        self.n = 0

    def score(self, X):
        """
        Compute outlier scores.

        Parameters
        ----------
        X
            Batch of instances to analyze.

        Returns
        -------
        Array with outlier scores for each instance in the batch.
        """
        n_batch, n_params = X.shape  # batch size and number of features
        n_components = min(self.n_components, n_params)
        if self.max_n is not None:
            n = min(self.n, self.max_n)  # n can never be above max_n
        else:
            n = self.n

        # clip X
        if self.n > self.start_clip:
            X_clip = np.clip(X, self.clip[0], self.clip[1])
        else:
            X_clip = X

        # track mean and covariance matrix
        roll_partial_means = X_clip.cumsum(axis=0) / (np.arange(n_batch) + 1).reshape((n_batch, 1))
        coefs = (np.arange(n_batch) + 1.) / (np.arange(n_batch) + n + 1.)
        new_means = self.mean + coefs.reshape((n_batch, 1)) * (roll_partial_means - self.mean)
        new_means_offset = np.empty_like(new_means)
        new_means_offset[0] = self.mean
        new_means_offset[1:] = new_means[:-1]

        coefs = ((n + np.arange(n_batch)) / (n + np.arange(n_batch) + 1.)).reshape((n_batch, 1, 1))
        B = coefs * np.matmul((X_clip - new_means_offset)[:, :, None], (X_clip - new_means_offset)[:, None, :])
        cov_batch = (n - 1.) / (n + max(1, n_batch - 1.)) * self.C + 1. / (n + max(1, n_batch - 1.)) * B.sum(axis=0)

        # PCA
        eigvals, eigvects = eigh(cov_batch, eigvals=(n_params - n_components, n_params - 1))

        # projections
        proj_x = np.matmul(X, eigvects)
        proj_x_clip = np.matmul(X_clip, eigvects)
        proj_means = np.matmul(new_means_offset, eigvects)
        if type(self.C) == int and self.C == 0:
            proj_cov = np.diag(np.zeros(n_components))
        else:
            proj_cov = np.matmul(eigvects.transpose(), np.matmul(self.C, eigvects))

        # outlier scores are computed in the principal component space
        coefs = (1. / (n + np.arange(n_batch) + 1.)).reshape((n_batch, 1, 1))
        B = coefs * np.matmul((proj_x_clip - proj_means)[:, :, None], (proj_x_clip - proj_means)[:, None, :])
        all_C_inv = np.zeros_like(B)
        c_inv = None
        for i, b in enumerate(B):
            if c_inv is None:
                if abs(np.linalg.det(proj_cov)) > EPSILON:
                    c_inv = np.linalg.inv(proj_cov)
                    all_C_inv[i] = c_inv
                    continue
                else:
                    if n + i == 0:
                        continue
                    proj_cov = (n + i - 1.) / (n + i) * proj_cov + b
                    continue
            else:
                c_inv = (n + i - 1.) / float(n + i - 2.) * all_C_inv[i - 1]
            BC1 = np.matmul(B[i - 1], c_inv)
            all_C_inv[i] = c_inv - 1. / (1. + np.trace(BC1)) * np.matmul(c_inv, BC1)

        # update parameters
        self.mean = new_means[-1]
        self.C = cov_batch
        stdev = np.sqrt(np.diag(cov_batch))
        self.n += n_batch
        if self.n > self.start_clip:
            self.clip = [self.mean - self.std_clip * stdev, self.mean + self.std_clip * stdev]

        # compute outlier scores
        x_diff = proj_x - proj_means
        scores = np.matmul(x_diff[:, None, :], np.matmul(all_C_inv, x_diff[:, :, None])).reshape(n_batch)
        return scores

    def predict(self, X):
        """
        Compute outlier scores and transform into outlier predictions.

        Parameters
        ----------
        X
            Batch of instances to analyze.

        Returns
        -------
        Array indicating which instances in the batch are outliers.
        """
        scores = self.score(X)  # compute outlier scores
        preds = (scores > self.threshold).astype(int)  # convert outlier scores into predictions
        return preds
