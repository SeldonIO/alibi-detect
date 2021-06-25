import numpy as np
import tensorflow as tf
from typing import Callable, Dict, Optional, Tuple, Union
from alibi_detect.cd.base import BaseLSDDDrift
from alibi_detect.utils.tensorflow.kernels import GaussianRBF
from alibi_detect.utils.tensorflow.distance import permed_lsdds


class LSDDDriftTF(BaseLSDDDrift):
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            p_val: float = .05,
            preprocess_x_ref: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            sigma: Optional[np.ndarray] = None,
            n_permutations: int = 100,
            n_kernel_centers: Optional[int] = None,
            lambda_rd_max: float = 0.2,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Least-squares density difference (LSDD) data drift detector using a permutation test.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        p_val
            p-value used for the significance of the permutation test.
        preprocess_x_ref
            Whether to already preprocess and store the reference data.
        update_x_ref
            Reference data can optionally be updated to the last n instances seen by the detector
            or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while
            for reservoir sampling {'reservoir_sampling': n} is passed.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        sigma
            Optionally set the bandwidth of the Gaussian kernel used in estimating the LSDD. Can also pass multiple
            bandwidth values as an array. The kernel evaluation is then averaged over those bandwidths. If `sigma`
            is not specified, the 'median heuristic' is adopted whereby `sigma` is set as the median pairwise distance
            between reference samples.
        n_permutations
            Number of permutations used in the permutation test.
        n_kernel_centers
            The number of reference samples to use as centers in the Gaussian kernel model used to estimate LSDD.
            Defaults to 1/20th of the reference data.
        lambda_rd_max
            The maximum relative difference between two estimates of LSDD that the regularization parameter
            lambda is allowed to cause. Defaults to 0.2 as in the paper.
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__(
            x_ref=x_ref,
            p_val=p_val,
            preprocess_x_ref=preprocess_x_ref,
            update_x_ref=update_x_ref,
            preprocess_fn=preprocess_fn,
            sigma=sigma,
            n_permutations=n_permutations,
            n_kernel_centers=n_kernel_centers,
            lambda_rd_max=lambda_rd_max,
            input_shape=input_shape,
            data_type=data_type
        )
        self.meta.update({'backend': 'tensorflow'})

        if self.preprocess_x_ref or self.preprocess_fn is None:
            x_ref = tf.convert_to_tensor(self.x_ref)
            self._configure_normalization(x_ref)
            x_ref = self._normalize(x_ref)
            self._initialize_kernel(x_ref)
            self._configure_kernel_centers(x_ref)
            self.x_ref = x_ref.numpy()
            # For stability in high dimensions we don't divide H by (pi*sigma^2)^(d/2)
            # Results in an alternative test-stat of LSDD*(pi*sigma^2)^(d/2). Same p-vals etc.
            self.H = GaussianRBF(np.sqrt(2.)*self.kernel.sigma)(self.kernel_centers, self.kernel_centers)

    def _initialize_kernel(self, x_ref: tf.Tensor):
        if self.sigma is None:
            self.kernel = GaussianRBF()
            _ = self.kernel(x_ref, x_ref, infer_sigma=True)
        else:
            sigma = tf.convert_to_tensor(self.sigma)
            self.kernel = GaussianRBF(sigma)

    def _configure_normalization(self, x_ref: tf.Tensor, eps: float = 1e-12):
        x_ref_means = tf.reduce_mean(x_ref, axis=0)
        x_ref_stds = tf.math.reduce_std(x_ref, axis=0)
        self._normalize = lambda x: (x - x_ref_means)/(x_ref_stds + eps)

    def _configure_kernel_centers(self, x_ref: tf.Tensor):
        "Set aside reference samples to act as kernel centers"
        perm = tf.random.shuffle(tf.range(self.x_ref.shape[0]))
        c_inds, non_c_inds = perm[:self.n_kernel_centers], perm[self.n_kernel_centers:]
        self.kernel_centers = tf.gather(x_ref, c_inds)
        if np.unique(self.kernel_centers.numpy(), axis=0).shape[0] < self.n_kernel_centers:
            perturbation = tf.random.normal(self.kernel_centers.shape, mean=0, stddev=1e-6)
            self.kernel_centers = self.kernel_centers + perturbation
        x_ref_eff = tf.gather(x_ref, non_c_inds)  # the effective reference set
        self.k_xc = self.kernel(x_ref_eff, self.kernel_centers)

    def score(self, x: Union[np.ndarray, list]) -> Tuple[float, float, np.ndarray]:
        """
        Compute the p-value resulting from a permutation test using the least-squares density
        difference as a distance measure between the reference data and the data to be tested.

        Parameters
        ----------
        x
            Batch of instances.

        Returns
        -------
        p-value obtained from the permutation test, the LSDD between the reference and test set
        and the LSDD values from the permutation test.
        """
        x_ref, x = self.preprocess(x)

        if self.preprocess_fn is not None and self.preprocess_x_ref is False:
            self._configure_normalization(x_ref)
            x_ref = self._normalize(x_ref)
            self._initialize_kernel(x_ref)
            self._configure_kernel_centers(x_ref)
            self.H = GaussianRBF(np.sqrt(2.)*self.kernel.sigma)(self.kernel_centers, self.kernel_centers)

        x = self._normalize(x)

        k_yc = self.kernel(x, self.kernel_centers)
        k_all_c = tf.concat([self.k_xc, k_yc], axis=0)

        n_x = x_ref.shape[0] - self.n_kernel_centers
        n_all = k_all_c.shape[0]
        perms = [tf.random.shuffle(tf.range(n_all)) for _ in range(self.n_permutations)]
        x_perms = [perm[:n_x] for perm in perms]
        y_perms = [perm[n_x:] for perm in perms]

        lsdd_permuted, _, lsdd = permed_lsdds(  # type: ignore
            k_all_c, x_perms, y_perms, self.H, lam_rd_max=self.lambda_rd_max, return_unpermed=True
        )

        p_val = tf.reduce_mean(tf.cast(lsdd <= lsdd_permuted, float))
        return float(p_val), float(lsdd), lsdd_permuted.numpy()
