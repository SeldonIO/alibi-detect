import logging
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from typing import Callable, Optional, Union
from alibi_detect.cd.base_online import BaseMMDDriftOnline
from alibi_detect.utils.tensorflow.kernels import GaussianRBF
from alibi_detect.cd.tensorflow.utils import zero_diag, quantile, subset_matrix

logger = logging.getLogger(__name__)


class MMDDriftOnlineTF(BaseMMDDriftOnline):
    def __init__(
            self,
            x_ref: np.ndarray,
            ert: float,
            window_size: int,
            preprocess_x_ref: bool = True,
            preprocess_fn: Optional[Callable] = None,
            kernel: Callable = GaussianRBF,
            sigma: Optional[np.ndarray] = None,
            n_bootstraps: int = 1000,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Online maximum Mean Discrepancy (MMD) data drift detector using preconfigured thresholds.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        ert
            The expected run-time (ERT) in the absence of drift.
        window_size
            The size of the sliding test-window used to compute the test-statistic.
            Smaller windows focus on responding quickly to severe drift, larger windows focus on
            ability to detect slight drift.
        preprocess_x_ref
            Whether to already preprocess and store the reference data.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        kernel
            Kernel used for the MMD computation, defaults to Gaussian RBF kernel.
        sigma
            Optionally set the GaussianRBF kernel bandwidth. Can also pass multiple bandwidth values as an array.
            The kernel evaluation is then averaged over those bandwidths.
        n_bootstraps
            The number of bootstrap simulations used to configure the thresholds. The larger this is the
            more accurately the desired ERT will be targeted. Should ideally be at least an order of magnitude
            larger than the ERT.
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__(
            x_ref=x_ref,
            ert=ert,
            window_size=window_size,
            preprocess_x_ref=preprocess_x_ref,
            preprocess_fn=preprocess_fn,
            sigma=sigma,
            n_bootstraps=n_bootstraps,
            input_shape=input_shape,
            data_type=data_type
        )
        self.meta.update({'backend': 'tensorflow'})

        # initialize kernel
        if isinstance(sigma, np.ndarray):
            sigma = tf.convert_to_tensor(sigma)
        self.kernel = kernel(sigma) if kernel == GaussianRBF else kernel

        # compute kernel matrix for the reference data
        self.k_xx = self.kernel(self.x_ref, self.x_ref, infer_sigma=(sigma is None))
        self.infer_sigma = False

        self._configure_thresholds()
        self._initialise()

    def _configure_ref_subset(self):
        etw_size = 2*self.window_size-1  # etw = extended test window
        rw_size = self.n - etw_size  # rw = ref window
        self.ref_inds = tf.random.shuffle(tf.range(self.n))[:-etw_size]
        self.k_xx_sub = subset_matrix(self.k_xx, self.ref_inds, self.ref_inds)
        self.k_xx_sub_sum = tf.reduce_sum(zero_diag(self.k_xx_sub))/(rw_size*(rw_size-1))

    def _configure_thresholds(self):

        # Each bootstrap sample splits the reference samples into a sub-reference sample (x)
        # and an extended test window (y). The extended test window will be treated as W overlapping
        # test windows of size W (so 2W-1 test samples in total)

        w_size = self.window_size
        etw_size = 2*w_size-1  # etw = extended test window
        rw_size = self.n - etw_size  # rw = ref window

        perms = [tf.random.shuffle(tf.range(self.n)) for _ in range(self.n_bootstraps)]
        x_inds_all = [perm[:-etw_size] for perm in perms]
        y_inds_all = [perm[-etw_size:] for perm in perms]

        print("Generating permutations of kernel matrix..")
        # Need to compute mmd for each bs for each of W overlapping windows
        # Most of the computation can be done once however
        # We avoid summing the rw_size^2 submatrix for each bootstrap sample by instead computing the full
        # sum once and then subtracting the relavent parts (k_xx_sum = k_full_sum - 2*k_xy_sum - k_yy_sum).
        # We also reduce computation of k_xy_sum from O(nW) to O(W) by caching column sums

        k_full_sum = tf.reduce_sum(zero_diag(self.k_xx))
        k_xy_col_sums_all = [
            tf.reduce_sum(subset_matrix(self.k_xx, x_inds, y_inds), axis=0) for x_inds, y_inds in
            tqdm(zip(x_inds_all, y_inds_all), total=self.n_bootstraps)
        ]
        k_xx_sums_all = [(
            k_full_sum -
            tf.reduce_sum(zero_diag(subset_matrix(self.k_xx, y_inds, y_inds))) -
            2*tf.reduce_sum(k_xy_col_sums)
        )/(rw_size*(rw_size-1)) for y_inds, k_xy_col_sums in zip(y_inds_all, k_xy_col_sums_all)]
        k_xy_col_sums_all = [k_xy_col_sums/(rw_size*w_size) for k_xy_col_sums in k_xy_col_sums_all]

        # Now to iterate through the W overlapping windows
        thresholds = []
        for w in tqdm(range(w_size), "Computing thresholds"):
            y_inds_all_w = [y_inds[w:w+w_size] for y_inds in y_inds_all]  # test windows of size W
            mmds = [(
                k_xx_sum +
                tf.reduce_sum(zero_diag(subset_matrix(self.k_xx, y_inds_w, y_inds_w)))/(w_size*(w_size-1)) -
                2*tf.reduce_sum(k_xy_col_sums[w:w+w_size])
            ) for k_xx_sum, y_inds_w, k_xy_col_sums in zip(k_xx_sums_all, y_inds_all_w, k_xy_col_sums_all)
            ]
            mmds = tf.concat(mmds, axis=0)  # an mmd for each bootstrap sample

            # Now we discard all bootstrap samples for which mmd is in top (1/ert)% and record the thresholds
            thresholds.append(quantile(mmds, 1-self.fpr))
            y_inds_all = [y_inds_all[i] for i in range(len(y_inds_all)) if mmds[i] < thresholds[-1]]
            k_xx_sums_all = [
                k_xx_sums_all[i] for i in range(len(k_xx_sums_all)) if mmds[i] < thresholds[-1]
            ]
            k_xy_col_sums_all = [
                k_xy_col_sums_all[i] for i in range(len(k_xy_col_sums_all)) if mmds[i] < thresholds[-1]
            ]

        self.thresholds = tf.concat(thresholds, axis=0)

    def score(self, x_t: np.ndarray) -> Union[float, None]:
        """
        Compute the test-statistic (squared MMD) between the reference window and test window.
        If the test-window is not yet full then a test-statistic of None is returned.

        Parameters
        ----------
        x_t
            A single instance.

        Returns
        -------
        Squared MMD estimate between reference window and test window.
        """
        x_t = x_t[None, :]
        kernel_col = self.kernel(self.x_ref[self.ref_inds], x_t)

        if self.t == 0:
            self.test_window = x_t
            self.k_xy = kernel_col
            return None
        elif 0 < self.t < self.window_size:
            self.test_window = tf.concat([self.test_window, x_t], axis=0)
            self.k_xy = tf.concat([self.k_xy, kernel_col], axis=1)
            return None
        elif self.t >= self.window_size:
            self.test_window = tf.concat([self.test_window[(1-self.window_size):], x_t], axis=0)
            self.k_xy = tf.concat([self.k_xy[:, (1-self.window_size):], kernel_col], axis=1)
            k_yy = self.kernel(self.test_window, self.test_window)
            mmd = (
                self.k_xx_sub_sum +
                tf.reduce_sum(zero_diag(k_yy))/(self.window_size*(self.window_size-1)) -
                2*tf.reduce_mean(self.k_xy)
            )
            return mmd.numpy()
