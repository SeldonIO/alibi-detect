import logging
from tqdm import tqdm
import numpy as np
import torch
from typing import Callable, Optional, Tuple
from alibi_detect.cd.base_online import BaseLSDDDriftOnline
from alibi_detect.utils.pytorch.kernels import GaussianRBF
from alibi_detect.cd.pytorch.utils import zero_diag, quantile

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
        device
            Device type used. The default None tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either 'cuda', 'gpu' or 'cpu'.
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

        # initialize kernel
        sigma = torch.from_numpy(sigma).to(self.device) if isinstance(sigma, np.ndarray) else None
        self.kernel = GaussianRBF(sigma)

        # compute kernel matrix for the reference data
        self.x_ref = torch.from_numpy(self.x_ref).to(self.device)
        _ = self.kernel(self.x_ref, self.x_ref, infer_sigma=(sigma is None))
        self.infer_sigma = False

        self._configure_kernel_centers()
        self._configure_thresholds()
        self._initialise()

    def _configure_kernel_centers(self):
        perm = torch.randperm(self.n)
        self.c_inds, self.non_c_inds = perm[:self.n_kernel_centers], perm[self.n_kernel_centers:]
        self.kernel_centers = self.x_ref[self.c_inds]
        self.x_ref_eff = self.x_ref[self.non_c_inds]  # the effective reference set
        self.k_xc = GaussianRBF(2*self.kernel.sigma)(self.x_ref_eff, self.kernel_centers)

    def _configure_ref_subset(self):
        etw_size = 2*self.window_size-1  # etw = extended test window
        nkc_size = self.n - self.n_kernel_centers  # nkc = non-kernel-centers
        rw_size = nkc_size - etw_size  # rw = ref-window
        self.ref_inds = torch.randperm(nkc_size)[:rw_size]
        self.c2s = self.k_xc[self.ref_inds].mean(0)

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

        x_ks = torch.stack([self.k_xc[x_inds] for x_inds in x_inds_all], axis=0)
        y_ks = torch.stack([self.k_xc[y_inds[:w_size]] for y_inds in y_inds_all], axis=0)
        hs = x_ks.mean(1) - y_ks.mean(1)

        eps = 1e-8
        candidate_lambdas = [1/(2**i)-eps for i in range(10)]
        H = GaussianRBF(4*self.kernel.sigma)(self.kernel_centers, self.kernel_centers) * \
            ((torch.tensor(np.pi)*self.kernel.sigma)**(d/2))
        H_plus_lams = torch.stack([H+torch.eye(H.shape[0])*can_lam for can_lam in candidate_lambdas], axis=0)
        H_plus_lam_invs = torch.inverse(H_plus_lams).permute(1, 2, 0)  # lambdas last

        omegas = torch.einsum('jkl,bk->bjl', H_plus_lam_invs, hs)

        h_omegas = torch.einsum('bj,bjl->bl', hs, omegas)
        omega_H_omegas = torch.einsum('bkl,bkl->bl', torch.einsum('bjl,jk->bkl', omegas, H), omegas)

        rds = (1 - (omega_H_omegas/h_omegas)).mean(0)
        lambda_index = (rds < self.lambda_rd_max).nonzero()[0]
        lam = candidate_lambdas[lambda_index]
        print(f"Using lambda value of {lam:.2g} with RD of {rds[lambda_index].item():.2g}")

        H_plus_lam_inv = H_plus_lam_invs[:, :, lambda_index.item()]
        self.H_lam_inv = 2*H_plus_lam_inv - (H_plus_lam_inv.transpose(0, 1) @ H @ H_plus_lam_inv)

        # distances = (2*h_omegas[:,lambda_index] - omega_H_omegas[:, lambda_index])[:,0].sort().values
        distances = (hs * (self.H_lam_inv @ hs.transpose(0, 1)).transpose(0, 1)).sum(-1)  # same thing

        self.thresholds = [quantile(torch.tensor(distances), 1-self.fpr)]
        for w in tqdm(range(1, w_size), "Computing thresholds"):
            x_ks = torch.stack([self.k_xc[x_inds] for x_inds in x_inds_all], axis=0)
            y_ks = torch.stack([self.k_xc[y_inds[w:(w+w_size)]] for y_inds in y_inds_all], axis=0)
            hs = x_ks.mean(1) - y_ks.mean(1)
            distances = (hs * (self.H_lam_inv @ hs.transpose(0, 1)).transpose(0, 1)).sum(-1)
            self.thresholds.append(quantile(torch.tensor(distances), 1-self.fpr))
            x_inds_all = [x_inds_all[i] for i in range(len(x_inds_all)) if distances[i] < self.thresholds[-1]]
            y_inds_all = [y_inds_all[i] for i in range(len(y_inds_all)) if distances[i] < self.thresholds[-1]]
        print(self.thresholds)

    def kernel_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ Compute and return full kernel matrix between arrays x and y. """
        k_xy = self.kernel(x, y, self.infer_sigma)
        k_xx = self.k_xx if self.k_xx is not None else self.kernel(x, x)
        k_yy = self.kernel(y, y)
        kernel_mat = torch.cat([torch.cat([k_xx, k_xy], 1), torch.cat([k_xy.T, k_yy], 1)], 0)
        return kernel_mat

    def score(self, x_t: np.ndarray) -> Tuple[float, float, np.ndarray]:
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
        x_t = torch.from_numpy(x_t[None, :]).to(self.device)
        k_xtc = GaussianRBF(2*self.kernel.sigma)(x_t, self.kernel_centers)

        if self.t == 0:
            self.test_window = x_t
            self.k_xtc = k_xtc
            return None
        elif 0 < self.t < self.window_size:
            self.test_window = torch.cat([self.test_window, x_t], axis=0)
            self.k_xtc = torch.cat([self.k_xtc, k_xtc], axis=0)
            return None
        elif self.t >= self.window_size:
            self.test_window = torch.cat([self.test_window[(1-self.window_size):], x_t], axis=0)
            self.k_xtc = torch.cat([self.k_xtc[(1-self.window_size):], k_xtc], axis=0)
            hs = self.c2s - self.k_xtc.mean(0)
            lsdd = hs[None, :] @ self.H_lam_inv @ hs[:, None]
            return float(lsdd.detach().cpu())
