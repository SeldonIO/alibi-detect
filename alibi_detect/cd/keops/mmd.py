import logging
import numpy as np
from pykeops.torch import LazyTensor
import torch
from typing import Callable, Dict, List, Optional, Tuple, Union
from alibi_detect.cd.base import BaseMMDDrift
from alibi_detect.utils.keops.kernels import GaussianRBF
from alibi_detect.utils.pytorch import get_device
from alibi_detect.utils.frameworks import Framework
from alibi_detect.utils._types import TorchDeviceType

logger = logging.getLogger(__name__)


class MMDDriftKeops(BaseMMDDrift):
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            p_val: float = .05,
            x_ref_preprocessed: bool = False,
            preprocess_at_init: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            kernel: Callable = GaussianRBF,
            sigma: Optional[np.ndarray] = None,
            configure_kernel_from_x_ref: bool = True,
            n_permutations: int = 100,
            batch_size_permutations: int = 1000000,
            device: TorchDeviceType = None,
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
        x_ref_preprocessed
            Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only
            the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference
            data will also be preprocessed.
        preprocess_at_init
            Whether to preprocess the reference data when the detector is instantiated. Otherwise, the reference
            data will be preprocessed at prediction time. Only applies if `x_ref_preprocessed=False`.
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
        batch_size_permutations
            KeOps computes the n_permutations of the MMD^2 statistics in chunks of batch_size_permutations.
        device
            Device type used. The default tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of
            ``torch.device``.
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__(
            x_ref=x_ref,
            p_val=p_val,
            x_ref_preprocessed=x_ref_preprocessed,
            preprocess_at_init=preprocess_at_init,
            update_x_ref=update_x_ref,
            preprocess_fn=preprocess_fn,
            sigma=sigma,
            configure_kernel_from_x_ref=configure_kernel_from_x_ref,
            n_permutations=n_permutations,
            input_shape=input_shape,
            data_type=data_type
        )
        self.meta.update({'backend': Framework.KEOPS.value})

        # set device
        self.device = get_device(device)

        # initialize kernel
        sigma = torch.from_numpy(sigma).to(self.device) if isinstance(sigma,  # type: ignore[assignment]
                                                                      np.ndarray) else None
        self.kernel = kernel(sigma).to(self.device) if kernel == GaussianRBF else kernel

        # set the correct MMD^2 function based on the batch size for the permutations
        self.batch_size = batch_size_permutations
        self.n_batches = 1 + (n_permutations - 1) // batch_size_permutations

        # infer the kernel bandwidth from the reference data
        if isinstance(sigma, torch.Tensor):
            self.infer_sigma = False
        elif self.infer_sigma:
            x = torch.from_numpy(self.x_ref).to(self.device)
            _ = self.kernel(LazyTensor(x[:, None, :]), LazyTensor(x[None, :, :]), infer_sigma=self.infer_sigma)
            self.infer_sigma = False
        else:
            self.infer_sigma = True

    def _mmd2(self, x_all: torch.Tensor, perms: List[torch.Tensor], m: int, n: int) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched (across the permutations) MMD^2 computation for the original test statistic and the permutations.

        Parameters
        ----------
        x_all
            Concatenated reference and test instances.
        perms
            List with permutation vectors.
        m
            Number of reference instances.
        n
            Number of test instances.

        Returns
        -------
        MMD^2 statistic for the original and permuted reference and test sets.
        """
        k_xx, k_yy, k_xy = [], [], []
        for batch in range(self.n_batches):
            i, j = batch * self.batch_size, (batch + 1) * self.batch_size
            # construct stacked tensors with a batch of permutations for the reference set x and test set y
            x = torch.cat([x_all[perm[:m]][None, :, :] for perm in perms[i:j]], 0)
            y = torch.cat([x_all[perm[m:]][None, :, :] for perm in perms[i:j]], 0)
            if batch == 0:
                x = torch.cat([x_all[None, :m, :], x], 0)
                y = torch.cat([x_all[None, m:, :], y], 0)
            x, y = x.to(self.device), y.to(self.device)

            # batch-wise kernel matrix computation over the permutations
            k_xy.append(self.kernel(
                LazyTensor(x[:, :, None, :]), LazyTensor(y[:, None, :, :]), self.infer_sigma).sum(1).sum(1).squeeze(-1))
            k_xx.append(self.kernel(
                LazyTensor(x[:, :, None, :]), LazyTensor(x[:, None, :, :])).sum(1).sum(1).squeeze(-1))
            k_yy.append(self.kernel(
                LazyTensor(y[:, :, None, :]), LazyTensor(y[:, None, :, :])).sum(1).sum(1).squeeze(-1))
        c_xx, c_yy, c_xy = 1 / (m * (m - 1)), 1 / (n * (n - 1)), 2. / (m * n)
        # Note that the MMD^2 estimates assume that the diagonal of the kernel matrix consists of 1's
        stats = c_xx * (torch.cat(k_xx) - m) + c_yy * (torch.cat(k_yy) - n) - c_xy * torch.cat(k_xy)
        return stats[0], stats[1:]

    def score(self, x: Union[np.ndarray, list]) -> Tuple[float, float, float]:
        """
        Compute the p-value resulting from a permutation test using the maximum mean discrepancy
        as a distance measure between the reference data and the data to be tested.

        Parameters
        ----------
        x
            Batch of instances.

        Returns
        -------
        p-value obtained from the permutation test, the MMD^2 between the reference and test set, \
        and the MMD^2 threshold above which drift is flagged.
        """
        x_ref, x = self.preprocess(x)
        x_ref = torch.from_numpy(x_ref).float()  # type: ignore[assignment]
        x = torch.from_numpy(x).float()  # type: ignore[assignment]
        # compute kernel matrix, MMD^2 and apply permutation test
        m, n = x_ref.shape[0], x.shape[0]
        perms = [torch.randperm(m + n) for _ in range(self.n_permutations)]
        # TODO - Rethink typings (related to https://github.com/SeldonIO/alibi-detect/issues/540)
        x_all = torch.cat([x_ref, x], 0)  # type: ignore[list-item]
        mmd2, mmd2_permuted = self._mmd2(x_all, perms, m, n)
        if self.device.type == 'cuda':
            mmd2, mmd2_permuted = mmd2.cpu(), mmd2_permuted.cpu()
        p_val = (mmd2 <= mmd2_permuted).float().mean()
        # compute distance threshold
        idx_threshold = int(self.p_val * len(mmd2_permuted))
        distance_threshold = torch.sort(mmd2_permuted, descending=True).values[idx_threshold]
        return p_val.numpy().item(), mmd2.numpy().item(), distance_threshold.numpy()
