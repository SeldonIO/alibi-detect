from tqdm import tqdm
import numpy as np
import tensorflow as tf
from typing import Any, Callable, Optional, Union
from alibi_detect.cd.base_online import BaseMultiDriftOnline
from alibi_detect.utils.tensorflow import GaussianRBF, quantile, permed_lsdds


class LSDDDriftOnlineTF(BaseMultiDriftOnline):
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            ert: float,
            window_size: int,
            preprocess_fn: Optional[Callable] = None,
            sigma: Optional[np.ndarray] = None,
            n_bootstraps: int = 1000,
            n_kernel_centers: Optional[int] = None,
            lambda_rd_max: float = 0.2,
            verbose: bool = True,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Online least squares density difference (LSDD) data drift detector using preconfigured thresholds.
        Motivated by Bu et al. (2017): https://ieeexplore.ieee.org/abstract/document/7890493
        Modifications are made such that a desired ERT can be accurately targeted however.

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
            Function to preprocess the data before computing the data drift metrics.s
        sigma
            Optionally set the bandwidth of the Gaussian kernel used in estimating the LSDD. Can also pass multiple
            bandwidth values as an array. The kernel evaluation is then averaged over those bandwidths. If `sigma`
            is not specified, the 'median heuristic' is adopted whereby `sigma` is set as the median pairwise distance
            between reference samples.
        n_bootstraps
            The number of bootstrap simulations used to configure the thresholds. The larger this is the
            more accurately the desired ERT will be targeted. Should ideally be at least an order of magnitude
            larger than the ert.
        n_kernel_centers
            The number of reference samples to use as centers in the Gaussian kernel model used to estimate LSDD.
            Defaults to 2*window_size.
        lambda_rd_max
            The maximum relative difference between two estimates of LSDD that the regularization parameter
            lambda is allowed to cause. Defaults to 0.2 as in the paper.
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
            n_bootstraps=n_bootstraps,
            verbose=verbose,
            input_shape=input_shape,
            data_type=data_type
        )
        self.meta.update({'backend': 'tensorflow'})
        self.n_kernel_centers = n_kernel_centers
        self.lambda_rd_max = lambda_rd_max

        self._configure_normalization()

        # initialize kernel
        if sigma is None:
            self.kernel = GaussianRBF()
            _ = self.kernel(self.x_ref, self.x_ref, infer_sigma=True)
        else:
            sigma = tf.convert_to_tensor(sigma)
            self.kernel = GaussianRBF(sigma)

        if self.n_kernel_centers is None:
            self.n_kernel_centers = 2*window_size

        self._configure_kernel_centers()
        self._configure_thresholds()
        self._initialise()

    def _configure_normalization(self, eps: float = 1e-12):
        x_ref_means = tf.reduce_mean(self.x_ref, axis=0)
        x_ref_stds = tf.math.reduce_std(self.x_ref, axis=0)
        self._normalize = lambda x: (x - x_ref_means)/(x_ref_stds + eps)
        self.x_ref = self._normalize(self.x_ref)

    def _configure_kernel_centers(self):
        "Set aside reference samples to act as kernel centers"
        perm = tf.random.shuffle(tf.range(self.n))
        self.c_inds, self.non_c_inds = perm[:self.n_kernel_centers], perm[self.n_kernel_centers:]
        self.kernel_centers = tf.gather(self.x_ref, self.c_inds)
        if np.unique(self.kernel_centers.numpy(), axis=0).shape[0] < self.n_kernel_centers:
            perturbation = tf.random.normal(self.kernel_centers.shape, mean=0, stddev=1e-6)
            self.kernel_centers = self.kernel_centers + perturbation
        self.x_ref_eff = tf.gather(self.x_ref, self.non_c_inds)  # the effective reference set
        self.k_xc = self.kernel(self.x_ref_eff, self.kernel_centers)

    def _configure_thresholds(self):

        # Each bootstrap sample splits the reference samples into a sub-reference sample (x)
        # and an extended test window (y). The extended test window will be treated as W overlapping
        # test windows of size W (so 2W-1 test samples in total)

        w_size = self.window_size
        etw_size = 2*w_size-1  # etw = extended test window
        nkc_size = self.n - self.n_kernel_centers  # nkc = non-kernel-centers
        rw_size = nkc_size - etw_size  # rw = ref-window

        perms = [tf.random.shuffle(tf.range(nkc_size)) for _ in range(self.n_bootstraps)]
        x_inds_all = [perm[:rw_size] for perm in perms]
        y_inds_all = [perm[rw_size:] for perm in perms]

        # For stability in high dimensions we don't divide H by (pi*sigma^2)^(d/2)
        # Results in an alternative test-stat of LSDD*(pi*sigma^2)^(d/2). Same p-vals etc.
        H = GaussianRBF(np.sqrt(2.)*self.kernel.sigma)(self.kernel_centers, self.kernel_centers)

        # Compute lsdds for first test-window. We infer regularisation constant lambda here.
        y_inds_all_0 = [y_inds[:w_size] for y_inds in y_inds_all]
        lsdds_0, H_lam_inv = permed_lsdds(
            self.k_xc, x_inds_all, y_inds_all_0, H, lam_rd_max=self.lambda_rd_max,
        )

        # Can compute threshold for first window
        thresholds = [quantile(lsdds_0, 1-self.fpr)]
        # And now to iterate through the other W-1 overlapping windows
        p_bar = tqdm(range(1, w_size), "Computing thresholds") if self.verbose else range(1, w_size)
        for w in p_bar:
            y_inds_all_w = [y_inds[w:(w+w_size)] for y_inds in y_inds_all]
            lsdds_w, _ = permed_lsdds(self.k_xc, x_inds_all, y_inds_all_w, H, H_lam_inv=H_lam_inv)
            thresholds.append(quantile(lsdds_w, 1-self.fpr))
            x_inds_all = [x_inds_all[i] for i in range(len(x_inds_all)) if lsdds_w[i] < thresholds[-1]]
            y_inds_all = [y_inds_all[i] for i in range(len(y_inds_all)) if lsdds_w[i] < thresholds[-1]]

        self.thresholds = thresholds
        self.H_lam_inv = H_lam_inv

    def _configure_ref_subset(self):
        etw_size = 2*self.window_size-1  # etw = extended test window
        nkc_size = self.n - self.n_kernel_centers  # nkc = non-kernel-centers
        rw_size = nkc_size - etw_size  # rw = ref-window
        # Make split and ensure it doesn't cause an initial detection
        lsdd_init = None
        while lsdd_init is None or lsdd_init >= self.get_threshold(0):
            # Make split
            perm = tf.random.shuffle(tf.range(nkc_size))
            self.ref_inds, self.init_test_inds = perm[:rw_size], perm[-self.window_size:]
            self.c2s = tf.reduce_mean(tf.gather(self.k_xc, self.ref_inds), axis=0)  # (below Eqn 21)
            self.test_window = tf.gather(self.x_ref_eff, self.init_test_inds)
            # Compute initial lsdd to check for initial detection
            self.k_xtc = self.kernel(self.test_window, self.kernel_centers)
            h_init = self.c2s - tf.reduce_mean(self.k_xtc, axis=0)  # (Eqn 21)
            lsdd_init = h_init[None, :] @ self.H_lam_inv @ h_init[:, None]  # (Eqn 11)

    def _update_state(self, x_t: tf.Tensor):
        self.t += 1
        k_xtc = self.kernel(x_t, self.kernel_centers)
        self.test_window = tf.concat([self.test_window[(1-self.window_size):], x_t], axis=0)
        self.k_xtc = tf.concat([self.k_xtc[(1-self.window_size):], k_xtc], axis=0)

    def score(self, x_t: Union[np.ndarray, Any]) -> float:
        """
        Compute the test-statistic (LSDD) between the reference window and test window.

        Parameters
        ----------
        x_t
            A single instance to be added to the test-window.

        Returns
        -------
        LSDD estimate between reference window and test window.
        """
        x_t = super()._preprocess_xt(x_t)
        x_t = tf.convert_to_tensor(x_t)
        x_t = self._normalize(x_t)
        self._update_state(x_t)
        h = self.c2s - tf.reduce_mean(self.k_xtc, axis=0)  # (Eqn 21)
        lsdd = h[None, :] @ self.H_lam_inv @ h[:, None]  # (Eqn 11)
        return float(lsdd)
