import logging
import numpy as np
import tensorflow as tf
from typing import Callable, Dict, Optional, Tuple, Union
from alibi_detect.cd.base import BaseMMDDrift
from alibi_detect.utils.tensorflow.distance import mmd2_from_kernel_matrix
from alibi_detect.utils.tensorflow.kernels import GaussianRBF

logger = logging.getLogger(__name__)


class MMDDriftTF(BaseMMDDrift):
    def __init__(
            self,
            x_ref: np.ndarray,
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
        self.kernel = kernel(sigma) if kernel == GaussianRBF else kernel

        # compute kernel matrix for the reference data
        if self.infer_sigma or isinstance(sigma, np.ndarray):
            self.k_xx = self.kernel(self.x_ref, self.x_ref, infer_sigma=configure_kernel_from_x_ref)
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

    def score(self, x: np.ndarray) -> Tuple[float, float, np.ndarray]:
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
             for _ in range(self.n_permutations)]
        )
        p_val = (mmd2 <= mmd2_permuted).mean()
        return p_val, mmd2, mmd2_permuted
