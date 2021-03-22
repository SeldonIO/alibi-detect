import logging
import numpy as np
import torch
from typing import Callable, Dict, Optional, Tuple
from alibi_detect.cd.base import BaseMMDDrift
from alibi_detect.utils.pytorch.distance import mmd2_from_kernel_matrix
from alibi_detect.utils.pytorch.kernels import GaussianRBF

logger = logging.getLogger(__name__)


class MMDDriftTorch(BaseMMDDrift):

    def __init__(
            self,
            x_ref: np.ndarray,
            p_val: float = .05,
            preprocess_x_ref: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            kernel: Callable = GaussianRBF,
            sigma: Optional[np.ndarray] = None,
            infer_sigma: bool = True,
            n_permutations: int = 100,
            device: Optional[str] = None,
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
            infer_sigma=infer_sigma,
            n_permutations=n_permutations,
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
        if self.infer_sigma or isinstance(sigma, torch.Tensor):
            x = torch.from_numpy(self.x_ref).to(self.device)
            self.k_xx = self.kernel(x, x, infer_sigma=infer_sigma)
            self.infer_sigma = False
        else:
            self.k_xx, self.infer_sigma = None, True

    def kernel_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ Compute and return full kernel matrix between arrays x and y. """
        k_xy = self.kernel(x, y, self.infer_sigma)
        k_xx = self.k_xx if self.k_xx is not None else self.kernel(x, x)
        k_yy = self.kernel(y, y)
        kernel_mat = torch.cat([torch.cat([k_xx, k_xy], 1), torch.cat([k_xy.T, k_yy], 1)], 0)
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
        x_ref = torch.from_numpy(x_ref).to(self.device)
        x = torch.from_numpy(x).to(self.device)
        # compute kernel matrix, MMD^2 and apply permutation test using the kernel matrix
        n = x.shape[0]
        kernel_mat = self.kernel_matrix(x_ref, x)
        kernel_mat = kernel_mat - torch.diag(kernel_mat.diag())  # zero diagonal
        mmd2 = mmd2_from_kernel_matrix(kernel_mat, n, permute=False, zero_diag=False)
        mmd2_permuted = torch.Tensor(
            [mmd2_from_kernel_matrix(kernel_mat, n, permute=True, zero_diag=False) for _ in range(self.n_permutations)]
        )
        if self.device.type == 'cuda':
            mmd2, mmd2_permuted = mmd2.cpu(), mmd2_permuted.cpu()
        p_val = (mmd2 <= mmd2_permuted).float().mean()
        return p_val.numpy().item(), mmd2.numpy().item(), mmd2_permuted.numpy()
