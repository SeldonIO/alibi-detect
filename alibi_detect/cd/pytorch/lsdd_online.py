import logging
from tqdm import tqdm
import numpy as np
import torch
from typing import Callable, Optional, Union
from alibi_detect.cd.base_online import BaseLSDDDriftOnline
from alibi_detect.utils.pytorch.kernels import GaussianRBF
from alibi_detect.utils.pytorch.distance import permed_lsdds
from alibi_detect.cd.pytorch.utils import quantile

logger = logging.getLogger(__name__)


class LSDDDriftOnlineTorch(BaseLSDDDriftOnline):
    def __init__(
            self,
            x_ref: np.ndarray,
            ert: float,
            window_size: int,
            preprocess_x_ref: bool = True,
            preprocess_fn: Optional[Callable] = None,
            sigma: Optional[np.ndarray] = None,
            n_bootstraps: int = 1000,
            n_kernel_centers: Optional[int] = None,
            lambda_rd_max: float = 0.2,
            device: Optional[str] = None,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Online least squares density difference (LSDD) data drift detector using preconfigured thresholds.
        Motivated by Bu et al. (2017): https://ieeexplore.ieee.org/abstract/document/7890493
        We have made modifications such that a desired ERT can be accurately targeted however.

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
            Function to preprocess the data before computing the data drift metrics.s
        sigma
            Optionally set the GaussianRBF kernel bandwidth. Can also pass multiple bandwidth values as an array.
            The kernel evaluation is then averaged over those bandwidths.
        n_bootstraps
            The number of bootstrap simulations used to configure the thresholds. The larger this is the
            more accurately the desired ERT will be targeted. Should ideally be at least an order of magnitude
            larger than the ert.
        n_kernel_centers
            Number of reference data points to use kernel centers to use in the estimation of the LSDD.
            Defaults to 2*window_size.
        lambda_rd_max
            The maximum relative difference between two estimates of LSDD that the regularization parameter
            lambda is allowed to cause. Defaults to 0.2 as in the paper.
        device
            Device type used. The default None tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either 'cuda', 'gpu' or 'cpu'. Only relevant for 'pytorch' backend.
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
            n_kernel_centers=n_kernel_centers,
            lambda_rd_max=lambda_rd_max,
            input_shape=input_shape,
            data_type=data_type
        )
        self.meta.update({'backend': 'pytorch'})

        # set backend
        if device is None or device.lower() in ['gpu', 'cuda']:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if self.device.type == 'cpu':
                print('No GPU detected, fall back on CPU.')
        else:
            self.device = torch.device('cpu')

        self._configure_normalization()

        # initialize kernel
        if sigma is None:
            self.x_ref = torch.from_numpy(self.x_ref).to(self.device)
            self.kernel = GaussianRBF()
            _ = self.kernel(self.x_ref, self.x_ref, infer_sigma=True)
        else:
            sigma = torch.from_numpy(sigma).to(self.device) if isinstance(sigma, np.ndarray) else None
            self.kernel = GaussianRBF(sigma)

        self._configure_kernel_centers()
        self._configure_thresholds()
        self._initialise()

    def _configure_normalization(self):
        x_ref_means = self.x_ref.mean(axis=0)
        x_ref_stds = self.x_ref.std(axis=0)
        self._normalize = lambda x: (x - x_ref_means)/(x_ref_stds)
        self.x_ref = self._normalize(self.x_ref)

    def _configure_kernel_centers(self):
        "Set aside reference samples to act as kernel centers"
        perm = torch.randperm(self.n)
        self.c_inds, self.non_c_inds = perm[:self.n_kernel_centers], perm[self.n_kernel_centers:]
        self.kernel_centers = self.x_ref[self.c_inds]
        self.x_ref_eff = self.x_ref[self.non_c_inds]  # the effective reference set
        self.k_xc = self.kernel(self.x_ref_eff, self.kernel_centers)

    def _configure_thresholds(self):

        # Each bootstrap sample splits the reference samples into a sub-reference sample (x)
        # and an extended test window (y). The extended test window will be treated as W overlapping
        # test windows of size W (so 2W-1 test samples in total)

        w_size = self.window_size
        etw_size = 2*w_size-1  # etw = extended test window
        nkc_size = self.n - self.n_kernel_centers  # nkc = non-kernel-centers
        rw_size = nkc_size - etw_size  # rw = ref-window
        d = self.x_ref_eff.shape[-1]

        perms = [torch.randperm(nkc_size) for _ in range(self.n_bootstraps)]
        x_inds_all = [perm[:rw_size] for perm in perms]
        y_inds_all = [perm[rw_size:] for perm in perms]

        H = GaussianRBF(np.sqrt(2.)*self.kernel.sigma)(self.kernel_centers, self.kernel_centers) * \
            ((torch.tensor(np.pi)*self.kernel.sigma**2)**(d/2))  # (Eqn 5)

        # Compute lsdds for first test-window. We infer regularisation constant lambda here.
        y_inds_all_0 = [y_inds[:w_size] for y_inds in y_inds_all]
        lsdds_0, H_lam_inv = permed_lsdds(
            self.k_xc, x_inds_all, y_inds_all_0, H, lam_rd_max=self.lambda_rd_max,
        )

        # Can compute threshold for first window
        thresholds = [quantile(lsdds_0, 1-self.fpr)]
        # And now to iterate through the other W-1 overlapping windows
        for w in tqdm(range(1, w_size), "Computing thresholds"):
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
            perm = torch.randperm(nkc_size)
            self.ref_inds, self.init_test_inds = perm[:rw_size], perm[-self.window_size:]
            self.c2s = self.k_xc[self.ref_inds].mean(0)  # (below Eqn 21)
            self.test_window = self.x_ref[self.init_test_inds]
            # Compute initial lsdd to check for initial detection
            self.k_xtc = self.kernel(self.test_window, self.kernel_centers)
            h_init = self.c2s - self.k_xtc.mean(0)  # (Eqn 21)
            lsdd_init = h_init[None, :] @ self.H_lam_inv @ h_init[:, None]  # (Eqn 11)

    def score(self, x_t: np.ndarray) -> Union[float, None]:
        """
        Compute the test-statistic (LSDD) between the reference window and test window.
        If the test-window is not yet full then a test-statistic of None is returned.

        Parameters
        ----------
        x_t
            A single instance.

        Returns
        -------
        LSDD estimate between reference window and test window.
        """
        x_t = torch.from_numpy(x_t[None, :]).to(self.device)
        x_t = self._normalize(x_t)
        k_xtc = self.kernel(x_t, self.kernel_centers)

        # if self.t == 0:
        #     self.test_window = x_t
        #     self.k_xtc = k_xtc
        #     return None
        # elif 0 < self.t < self.window_size:
        #     self.test_window = torch.cat([self.test_window, x_t], 0)
        #     self.k_xtc = torch.cat([self.k_xtc, k_xtc], 0)
        #     return None
        # else:
        self.test_window = torch.cat([self.test_window[(1-self.window_size):], x_t], 0)
        self.k_xtc = torch.cat([self.k_xtc[(1-self.window_size):], k_xtc], 0)
        h = self.c2s - self.k_xtc.mean(0)  # (Eqn 21)
        lsdd = h[None, :] @ self.H_lam_inv @ h[:, None]  # (Eqn 11)
        return float(lsdd.detach().cpu())
