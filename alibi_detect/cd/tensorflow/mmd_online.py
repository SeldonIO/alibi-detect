import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from typing import Any, Callable, Optional, Union
from alibi_detect.cd.base_online import BaseMultiDriftOnline
from alibi_detect.utils.tensorflow.kernels import GaussianRBF
from alibi_detect.utils.tensorflow import zero_diag, quantile, subset_matrix
from alibi_detect.utils.frameworks import Framework


class MMDDriftOnlineTF(BaseMultiDriftOnline):
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            ert: float,
            window_size: int,
            preprocess_fn: Optional[Callable] = None,
            x_ref_preprocessed: bool = False,
            kernel: Callable = GaussianRBF,
            sigma: Optional[np.ndarray] = None,
            n_bootstraps: int = 1000,
            verbose: bool = True,
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
            The expected run-time (ERT) in the absence of drift. For the multivariate detectors, the ERT is defined
            as the expected run-time from t=0.
        window_size
            The size of the sliding test-window used to compute the test-statistic.
            Smaller windows focus on responding quickly to severe drift, larger windows focus on
            ability to detect slight drift.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        x_ref_preprocessed
            Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only
            the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference
            data will also be preprocessed.
        kernel
            Kernel used for the MMD computation, defaults to Gaussian RBF kernel.
        sigma
            Optionally set the GaussianRBF kernel bandwidth. Can also pass multiple bandwidth values as an array.
            The kernel evaluation is then averaged over those bandwidths. If `sigma` is not specified, the 'median
            heuristic' is adopted whereby `sigma` is set as the median pairwise distance between reference samples.
        n_bootstraps
            The number of bootstrap simulations used to configure the thresholds. The larger this is the
            more accurately the desired ERT will be targeted. Should ideally be at least an order of magnitude
            larger than the ERT.
        verbose
            Whether or not to print progress during configuration.
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__(
            x_ref=x_ref,
            ert=ert,
            window_size=window_size,
            preprocess_fn=preprocess_fn,
            x_ref_preprocessed=x_ref_preprocessed,
            n_bootstraps=n_bootstraps,
            verbose=verbose,
            input_shape=input_shape,
            data_type=data_type
        )
        self.meta.update({'backend': Framework.TENSORFLOW.value})

        # initialize kernel
        if isinstance(sigma, np.ndarray):
            sigma = tf.convert_to_tensor(sigma)
        self.kernel = kernel(sigma) if kernel == GaussianRBF else kernel

        # compute kernel matrix for the reference data
        self.k_xx = self.kernel(self.x_ref, self.x_ref, infer_sigma=(sigma is None))

        self._configure_thresholds()
        self._configure_ref_subset()  # self.initialise_state() called inside here

    def _initialise_state(self) -> None:
        """
        Initialise online state (the stateful attributes updated by `score` and `predict`). This method relies on
        attributes defined by `_configure_ref_subset`, hence must be called afterwards.
        """
        super()._initialise_state()
        self.test_window = tf.gather(self.x_ref, self.init_test_inds)
        self.k_xy = self.kernel(tf.gather(self.x_ref, self.ref_inds), self.test_window)

    def _configure_ref_subset(self):
        """
        Configure the reference data split. If the randomly selected split causes an initial detection, further splits
        are attempted.
        """
        etw_size = 2 * self.window_size - 1  # etw = extended test window
        rw_size = self.n - etw_size  # rw = ref window#
        # Make split and ensure it doesn't cause an initial detection
        mmd_init = None
        while mmd_init is None or mmd_init >= self.get_threshold(0):
            # Make split
            perm = tf.random.shuffle(tf.range(self.n))
            self.ref_inds, self.init_test_inds = perm[:rw_size], perm[-self.window_size:]
            # Compute initial mmd to check for initial detection
            self._initialise_state()  # to set self.test_window and self.k_xtc
            self.k_xx_sub = subset_matrix(self.k_xx, self.ref_inds, self.ref_inds)
            self.k_xx_sub_sum = tf.reduce_sum(zero_diag(self.k_xx_sub)) / (rw_size * (rw_size - 1))
            k_yy = self.kernel(self.test_window, self.test_window)
            mmd_init = (
                    self.k_xx_sub_sum +
                    tf.reduce_sum(zero_diag(k_yy)) / (self.window_size * (self.window_size - 1)) -
                    2 * tf.reduce_mean(self.k_xy)
            )

    def _configure_thresholds(self):
        """
        Configure the test statistic thresholds via bootstrapping.
        """
        # Each bootstrap sample splits the reference samples into a sub-reference sample (x)
        # and an extended test window (y). The extended test window will be treated as W overlapping
        # test windows of size W (so 2W-1 test samples in total)

        w_size = self.window_size
        etw_size = 2 * w_size - 1  # etw = extended test window
        rw_size = self.n - etw_size  # rw = ref window

        perms = [tf.random.shuffle(tf.range(self.n)) for _ in range(self.n_bootstraps)]
        x_inds_all = [perm[:-etw_size] for perm in perms]
        y_inds_all = [perm[-etw_size:] for perm in perms]

        if self.verbose:
            print("Generating permutations of kernel matrix..")
        # Need to compute mmd for each bs for each of W overlapping windows
        # Most of the computation can be done once however
        # We avoid summing the rw_size^2 submatrix for each bootstrap sample by instead computing the full
        # sum once and then subtracting the relavent parts (k_xx_sum = k_full_sum - 2*k_xy_sum - k_yy_sum).
        # We also reduce computation of k_xy_sum from O(nW) to O(W) by caching column sums

        k_full_sum = tf.reduce_sum(zero_diag(self.k_xx))
        k_xy_col_sums_all = [
            tf.reduce_sum(subset_matrix(self.k_xx, x_inds, y_inds), axis=0) for x_inds, y_inds in
            (tqdm(zip(x_inds_all, y_inds_all), total=self.n_bootstraps) if self.verbose else
             zip(x_inds_all, y_inds_all))
        ]
        k_xx_sums_all = [(
                                 k_full_sum -
                                 tf.reduce_sum(zero_diag(subset_matrix(self.k_xx, y_inds, y_inds))) -
                                 2 * tf.reduce_sum(k_xy_col_sums)
                         ) / (rw_size * (rw_size - 1)) for y_inds, k_xy_col_sums in zip(y_inds_all, k_xy_col_sums_all)]
        k_xy_col_sums_all = [k_xy_col_sums / (rw_size * w_size) for k_xy_col_sums in k_xy_col_sums_all]

        # Now to iterate through the W overlapping windows
        thresholds = []
        p_bar = tqdm(range(w_size), "Computing thresholds") if self.verbose else range(w_size)
        for w in p_bar:
            y_inds_all_w = [y_inds[w:w + w_size] for y_inds in y_inds_all]  # test windows of size W
            mmds = [(
                    k_xx_sum +
                    tf.reduce_sum(zero_diag(subset_matrix(self.k_xx, y_inds_w, y_inds_w))) / (w_size * (w_size - 1)) -
                    2 * tf.reduce_sum(k_xy_col_sums[w:w + w_size]))
                    for k_xx_sum, y_inds_w, k_xy_col_sums in zip(k_xx_sums_all, y_inds_all_w, k_xy_col_sums_all)
                    ]
            mmds = tf.concat(mmds, axis=0)  # an mmd for each bootstrap sample

            # Now we discard all bootstrap samples for which mmd is in top (1/ert)% and record the thresholds
            thresholds.append(quantile(mmds, 1 - self.fpr))
            y_inds_all = [y_inds_all[i] for i in range(len(y_inds_all)) if mmds[i] < thresholds[-1]]
            k_xx_sums_all = [
                k_xx_sums_all[i] for i in range(len(k_xx_sums_all)) if mmds[i] < thresholds[-1]
            ]
            k_xy_col_sums_all = [
                k_xy_col_sums_all[i] for i in range(len(k_xy_col_sums_all)) if mmds[i] < thresholds[-1]
            ]

        self.thresholds = thresholds

    def _update_state(self, x_t: np.ndarray):  # type: ignore[override]
        """
        Update online state based on the provided test instance.

        Parameters
        ----------
        x_t
            The test instance.
        """
        self.t += 1
        kernel_col = self.kernel(self.x_ref[self.ref_inds], x_t)
        self.test_window = tf.concat([self.test_window[(1 - self.window_size):], x_t], axis=0)
        self.k_xy = tf.concat([self.k_xy[:, (1 - self.window_size):], kernel_col], axis=1)

    def score(self, x_t: Union[np.ndarray, Any]) -> float:
        """
        Compute the test-statistic (squared MMD) between the reference window and test window.

        Parameters
        ----------
        x_t
            A single instance to be added to the test-window.

        Returns
        -------
        Squared MMD estimate between reference window and test window.
        """
        x_t = super()._preprocess_xt(x_t)
        self._update_state(x_t)
        k_yy = self.kernel(self.test_window, self.test_window)
        mmd = (
                self.k_xx_sub_sum +
                tf.reduce_sum(zero_diag(k_yy)) / (self.window_size * (self.window_size - 1)) -
                2 * tf.reduce_mean(self.k_xy)
        )
        return mmd.numpy()

    def save_state(self, filepath: Union[str, os.PathLike]):
        """
        Save a detector's state to disk in order to generate a checkpoint.

        Parameters
        ----------
        filepath
            The directory to save state to.
        """
        super()._set_state_path(filepath)
        state_dict = {
            't': self.t,
            'test_window': self.test_window,
            'k_xy': self.k_xy
        }
        np.savez(self.state_path.joinpath('state.npz'), **state_dict)

    def load_state(self, filepath: Union[str, os.PathLike]):
        """
        Load the detector's state from disk, in order to restart from a checkpoint previously generated with
        `save_state`.

        Parameters
        ----------
        filepath
            The directory to load state from.
        """
        super()._set_state_path(filepath)
        state_dict = np.load(self.state_path.joinpath('state.npz'))
        self.t = state_dict.get('t')
        self.test_window = state_dict.get('test_window')
        self.k_xy = state_dict.get('k_xy')
