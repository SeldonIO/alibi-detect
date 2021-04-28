import logging
from tqdm import tqdm
import numpy as np
import torch
from typing import Callable, Dict, Optional, Tuple
from alibi_detect.cd.base_online import BaseMMDDriftOnline
from alibi_detect.utils.pytorch.distance import mmd2_from_kernel_matrix
from alibi_detect.utils.pytorch.kernels import GaussianRBF
from alibi_detect.cd.pytorch.utils import zero_diag, quantile

logger = logging.getLogger(__name__)


class MMDDriftOnlineTorch(BaseMMDDriftOnline):
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
        self.kernel = kernel(sigma) if kernel == GaussianRBF else kernel

        # compute kernel matrix for the reference data
        self.x_ref = torch.from_numpy(self.x_ref).to(self.device)
        self.k_xx = self.kernel(self.x_ref, self.x_ref, infer_sigma=(sigma is None))
        self.infer_sigma = False

        self._initialise()
        self._configure_thresholds()

    def _configure_ref_subset(self):
        self.ref_inds = torch.randperm(self.n)[:(-2*self.window_size)]
        self.k_xx_sub = self.k_xx[self.ref_inds][:, self.ref_inds]
        self.k_xx_sub_sum = zero_diag(self.k_xx_sub).sum()/(len(self.ref_inds)*(len(self.ref_inds)-1))

    def _configure_thresholds(self):

        perms = [torch.randperm(self.n) for _ in range(self.n_bootstraps)]
        p_inds_all = [perm[:(-2*self.window_size)] for perm in perms]
        q_inds_all = [perm[(-2*self.window_size):] for perm in perms]

        thresholds = []

        rw_size = self.n - 2*self.window_size

        print("Generating permutations of kernel matrix..")
        k_xy_col_sums_all = [
            self.k_xx[p_inds][:, q_inds].sum(0) for p_inds, q_inds in \
            tqdm(zip(p_inds_all, q_inds_all), total=self.n_bootstraps)
        ]
        k_full_sum = zero_diag(self.k_xx).sum()
        k_xx_sums_all = [(
            k_full_sum - zero_diag(self.k_xx[q_inds][:,q_inds]).sum() - 2*k_xy_col_sums.sum()
        )/(rw_size*(rw_size-1)) for q_inds, k_xy_col_sums in zip(q_inds_all, k_xy_col_sums_all)]  # This is bottleneck w.r.t. large num_bootstraps
        k_xy_col_sums_all = [k_xy_col_sums/(rw_size*self.window_size) for k_xy_col_sums in k_xy_col_sums_all]

        for w in tqdm(range(self.window_size), "Computing thresholds"):
            q_inds_all_w = [q_inds[w:w+self.window_size] for q_inds in q_inds_all]
            mmds = [(
                k_xx_sum +
                zero_diag(self.k_xx[q_inds_w][:, q_inds_w]).sum()/(self.window_size*(self.window_size-1)) -
                2*k_xy_col_sums[w:w+self.window_size].sum()
            ) for k_xx_sum, q_inds_w, k_xy_col_sums in zip(k_xx_sums_all, q_inds_all_w, k_xy_col_sums_all)
            ]

            mmds = torch.tensor(mmds)

            thresholds.append(quantile(mmds, 1-self.fpr))
            q_inds_all = [q_inds_all[i] for i in range(len(q_inds_all)) if mmds[i] < thresholds[-1]]
            k_xx_sums_all = [
                k_xx_sums_all[i] for i in range(len(k_xx_sums_all)) if mmds[i] < thresholds[-1]
            ]
            k_xy_col_sums_all = [
                k_xy_col_sums_all[i] for i in range(len(k_xy_col_sums_all)) if mmds[i] < thresholds[-1]
            ]
  
        self.thresholds = torch.stack(thresholds, axis=0).detach().cpu().numpy()

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
        kernel_col = self.kernel(self.x_ref[self.ref_inds], x_t)
        if self.t == 0:
            self.test_window = x_t
            self.k_xy = kernel_col
            return None
        elif 0 < self.t < self.window_size:
            self.test_window = torch.cat([self.test_window, x_t], axis=0)
            self.k_xy = torch.cat([self.k_xy, kernel_col], axis=1)
            return None
        elif self.t >= self.window_size:
            self.test_window = torch.cat([self.test_window[(1-self.window_size):], x_t], axis=0)
            self.k_xy = torch.cat([self.k_xy[:, (1-self.window_size):], kernel_col], axis=1)
            k_yy = self.kernel(self.test_window, self.test_window)
            mmd = (
                self.k_xx_sub_sum +
                zero_diag(k_yy).sum()/(self.window_size*(self.window_size-1)) -
                2*self.k_xy.mean()
            )
            return float(mmd.detach().cpu())
