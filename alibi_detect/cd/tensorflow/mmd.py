import logging
import numpy as np
import scipy.stats as stats
import tensorflow as tf
from typing import Callable, Dict, Optional, Tuple, Union
from alibi_detect.cd.base import BaseMMDDrift
from alibi_detect.utils.tensorflow.distance import mmd2_from_kernel_matrix, linear_mmd2
from alibi_detect.utils.tensorflow.kernels import GaussianRBF

logger = logging.getLogger(__name__)


class MMDDriftTF(BaseMMDDrift):
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            p_val: float = .05,
            preprocess_x_ref: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            kernel: Callable = GaussianRBF,
            sigma: Optional[np.ndarray] = None,
            configure_kernel_from_x_ref: bool = True,
            n_permutations: int = 100,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Maximum Mean Discrepancy (MMD) data drift detector using a permutation test.

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
        kernel
            Kernel used for the MMD computation, defaults to Gaussian RBF kernel.
        sigma
            Optionally set the GaussianRBF kernel bandwidth. Can also pass multiple bandwidth values as an array.
            The kernel evaluation is then averaged over those bandwidths.
        configure_kernel_from_x_ref
            Whether to already configure the kernel bandwidth from the reference data.
        n_permutations
            Number of permutations used in the permutation test.
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
            configure_kernel_from_x_ref=configure_kernel_from_x_ref,
            n_permutations=n_permutations,
            input_shape=input_shape,
            data_type=data_type
        )
        self.meta.update({'backend': 'tensorflow'})

        # initialize kernel
        if isinstance(sigma, np.ndarray):
            sigma = tf.convert_to_tensor(sigma)
        self.kernel = kernel(sigma) if kernel == GaussianRBF else kernel

        # compute kernel matrix for the reference data
        if self.infer_sigma or isinstance(sigma, tf.Tensor):
            self.k_xx = self.kernel(self.x_ref, self.x_ref, infer_sigma=self.infer_sigma)
            self.infer_sigma = False
        else:
            self.k_xx, self.infer_sigma = None, True

    def kernel_matrix(self, x: Union[np.ndarray, tf.Tensor], y: Union[np.ndarray, tf.Tensor]) -> tf.Tensor:
        """ Compute and return full kernel matrix between arrays x and y. """
        k_xy = self.kernel(x, y, self.infer_sigma)
        k_xx = self.k_xx if self.k_xx is not None and self.update_x_ref is None else self.kernel(x, x)
        k_yy = self.kernel(y, y)
        kernel_mat = tf.concat([tf.concat([k_xx, k_xy], 1), tf.concat([tf.transpose(k_xy, (1, 0)), k_yy], 1)], 0)
        return kernel_mat

    def score(self, x: Union[np.ndarray, list]) -> Tuple[float, float, np.ndarray]:
        """
        Compute the p-value resulting from a permutation test using the maximum mean discrepancy
        as a distance measure between the reference data and the data to be tested.

        Parameters
        ----------
        x
            Batch of instances.

        Returns
        -------
        p-value obtained from the permutation test, the MMD^2 between the reference and test set
        and the MMD^2 values from the permutation test.
        """
        x_ref, x = self.preprocess(x)
        # compute kernel matrix, MMD^2 and apply permutation test using the kernel matrix
        n = x.shape[0]
        kernel_mat = self.kernel_matrix(x_ref, x)
        kernel_mat = kernel_mat - tf.linalg.diag(tf.linalg.diag_part(kernel_mat))  # zero diagonal
        mmd2 = mmd2_from_kernel_matrix(kernel_mat, n, permute=False, zero_diag=False).numpy()
        mmd2_permuted = np.array(
            [mmd2_from_kernel_matrix(kernel_mat, n, permute=True, zero_diag=False).numpy()
             for _ in range(self.n_permutations)])
        p_val = (mmd2 <= mmd2_permuted).mean()
        return p_val, mmd2, mmd2_permuted


class LinearTimeMMDDriftTF(BaseMMDDrift):
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            p_val: float = .05,
            preprocess_x_ref: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            kernel: Callable = GaussianRBF,
            sigma: Optional[np.ndarray] = None,
            configure_kernel_from_x_ref: bool = True,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Maximum Mean Discrepancy (MMD) data drift detector using a linear-time estimator.

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
        kernel
            Kernel used for the MMD computation, defaults to Gaussian RBF kernel.
        sigma
            Optionally set the GaussianRBF kernel bandwidth. Can also pass multiple bandwidth values as an array.
            The kernel evaluation is then averaged over those bandwidths.
        configure_kernel_from_x_ref
            Whether to already configure the kernel bandwidth from the reference data.
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
            configure_kernel_from_x_ref=configure_kernel_from_x_ref,
            input_shape=input_shape,
            data_type=data_type
        )
        self.meta.update({'backend': 'tensorflow'})

        # initialize kernel
        if isinstance(sigma, np.ndarray):
            sigma = tf.convert_to_tensor(sigma)
        self.kernel = kernel(sigma) if kernel == GaussianRBF else kernel

        # compute kernel matrix for the reference data
        if self.infer_sigma or isinstance(sigma, tf.Tensor):
            n = self.x_ref.shape[0]
            n_hat = int(np.floor(n / 2) * 2)
            x = self.x_ref[:n_hat, :]
            self.k_xx = self.kernel(x=x[0::2, :], y=x[1::2, :],
                                    pairwise=False, infer_sigma=self.infer_sigma)
            self.infer_sigma = False
        else:
            self.k_xx, self.infer_sigma = None, True

    def kernel_matrix(self, x: Union[np.ndarray, tf.Tensor], y: Union[np.ndarray, tf.Tensor]) -> tf.Tensor:
        """ Compute and return full kernel matrix between arrays x and y. """
        k_xy = self.kernel(x, y, self.infer_sigma)
        k_xx = self.k_xx if self.k_xx is not None and self.update_x_ref is None else self.kernel(x, x)
        k_yy = self.kernel(y, y)
        kernel_mat = tf.concat([tf.concat([k_xx, k_xy], 1), tf.concat([tf.transpose(k_xy, (1, 0)), k_yy], 1)], 0)
        return kernel_mat

    def score(self, x: Union[np.ndarray, list]) -> Tuple[float, float, np.ndarray]:
        """
        Compute the p-value using the maximum mean discrepancy as a distance measure between the
        reference data and the data to be tested. The sample size is specified as the maximal even
        number in the smaller dataset between x and x_ref.

        Parameters
        ----------
        x
            Batch of instances.

        Returns
        -------
        p-value obtained from the null hypothesis, the MMD^2 between the reference and test set
        and the MMD^2 threshold for the given significance level.
        """
        x_ref, x = self.preprocess(x)
        # compute kernel matrix, MMD^2 and apply permutation test using the kernel matrix
        n = x.shape[0]
        m = x_ref.shape[0]
        if n != m:
            raise ValueError('x and x_ref must have the same size.')
        n_hat = int(np.floor(n / 2) * 2)
        x_ref = x_ref[:n_hat, :]
        x = x[:n_hat, :]
        if self.k_xx is not None and self.update_x_ref is None:
            k_xx = self.k_xx
        else:
            k_xx = self.kernel(x=x_ref[0::2, :], y=x_ref[1::2, :], pairwise=False)
        mmd2, var_mmd2 = linear_mmd2(k_xx, x_ref, x, self.kernel)
        mmd2 = mmd2.numpy()
        var_mmd2 = np.clip(var_mmd2.numpy(), 1e-8, 1e8)
        std_mmd2 = np.sqrt(var_mmd2)
        t = mmd2 / (std_mmd2 / np.sqrt(n_hat / 2.))
        p_val = 1 - stats.t.cdf(t, df=(n_hat / 2.) - 1)
        distance_threshold = stats.t.ppf(1 - self.p_val, df=(n_hat / 2.) - 1)
        return p_val, t, distance_threshold
