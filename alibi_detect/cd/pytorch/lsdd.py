import numpy as np
import torch
from typing import Callable, Dict, Optional, Tuple, Union
from alibi_detect.cd.base import BaseLSDDDrift
from alibi_detect.utils.pytorch import get_device
from alibi_detect.utils.pytorch.kernels import GaussianRBF
from alibi_detect.utils.pytorch.distance import permed_lsdds
from alibi_detect.utils.warnings import deprecated_alias
from alibi_detect.utils.frameworks import Framework
from alibi_detect.utils._types import TorchDeviceType


class LSDDDriftTorch(BaseLSDDDrift):
    @deprecated_alias(preprocess_x_ref='preprocess_at_init')
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            p_val: float = .05,
            x_ref_preprocessed: bool = False,
            preprocess_at_init: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            sigma: Optional[np.ndarray] = None,
            n_permutations: int = 100,
            n_kernel_centers: Optional[int] = None,
            lambda_rd_max: float = 0.2,
            device: TorchDeviceType = None,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Least-squares density difference (LSDD) data drift detector using a permutation test.

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
        sigma
            Optionally set the bandwidth of the Gaussian kernel used in estimating the LSDD. Can also pass multiple
            bandwidth values as an array. The kernel evaluation is then averaged over those bandwidths. If `sigma`
            is not specified, the 'median heuristic' is adopted whereby `sigma` is set as the median pairwise distance
            between reference samples.
        n_permutations
            Number of permutations used in the permutation test.
        n_kernel_centers
            The number of reference samples to use as centers in the Gaussian kernel model used to estimate LSDD.
            Defaults to 1/20th of the reference data.
        lambda_rd_max
            The maximum relative difference between two estimates of LSDD that the regularization parameter
            lambda is allowed to cause. Defaults to 0.2 as in the paper.
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
            n_permutations=n_permutations,
            n_kernel_centers=n_kernel_centers,
            lambda_rd_max=lambda_rd_max,
            input_shape=input_shape,
            data_type=data_type
        )
        self.meta.update({'backend': Framework.PYTORCH.value})

        # set device
        self.device = get_device(device)

        # TODO: TBD: the several type:ignore's below are because x_ref is typed as an np.ndarray
        #  in the method signature, so we can't cast it to torch.Tensor unless we change the signature
        #  to also accept torch.Tensor. We also can't redefine it's type as that would involve enabling
        #  --allow-redefinitions in mypy settings (which we might do eventually).
        if self.preprocess_at_init or self.preprocess_fn is None or self.x_ref_preprocessed:
            x_ref = torch.as_tensor(self.x_ref).to(self.device)  # type: ignore[assignment]
            self._configure_normalization(x_ref)  # type: ignore[arg-type]
            x_ref = self._normalize(x_ref)
            self._initialize_kernel(x_ref)  # type: ignore[arg-type]
            self._configure_kernel_centers(x_ref)  # type: ignore[arg-type]
            self.x_ref = x_ref.cpu().numpy()  # type: ignore[union-attr]
            # For stability in high dimensions we don't divide H by (pi*sigma^2)^(d/2)
            # Results in an alternative test-stat of LSDD*(pi*sigma^2)^(d/2). Same p-vals etc.
            self.H = GaussianRBF(np.sqrt(2.) * self.kernel.sigma)(self.kernel_centers, self.kernel_centers)

    def _initialize_kernel(self, x_ref: torch.Tensor):
        if self.sigma is None:
            self.kernel = GaussianRBF()
            _ = self.kernel(x_ref, x_ref, infer_sigma=True)
        else:
            sigma = torch.from_numpy(self.sigma)
            self.kernel = GaussianRBF(sigma)

    def _configure_normalization(self, x_ref: torch.Tensor, eps: float = 1e-12):
        x_ref_means = x_ref.mean(0)
        x_ref_stds = x_ref.std(0)
        self._normalize = lambda x: (torch.as_tensor(x) - x_ref_means) / (x_ref_stds + eps)
        self._unnormalize = lambda x: (torch.as_tensor(x) * (x_ref_stds + eps)
                                       + x_ref_means).cpu().numpy()

    def _configure_kernel_centers(self, x_ref: torch.Tensor):
        "Set aside reference samples to act as kernel centers"
        perm = torch.randperm(self.x_ref.shape[0])
        c_inds, non_c_inds = perm[:self.n_kernel_centers], perm[self.n_kernel_centers:]
        self.kernel_centers = x_ref[c_inds]
        if np.unique(self.kernel_centers.cpu().numpy(), axis=0).shape[0] < self.n_kernel_centers:
            perturbation = (torch.randn(self.kernel_centers.shape) * 1e-6).to(self.device)
            self.kernel_centers = self.kernel_centers + perturbation
        x_ref_eff = x_ref[non_c_inds]  # the effective reference set
        self.k_xc = self.kernel(x_ref_eff, self.kernel_centers)

    def score(self, x: Union[np.ndarray, list]) -> Tuple[float, float, float]:
        """
        Compute the p-value resulting from a permutation test using the least-squares density
        difference as a distance measure between the reference data and the data to be tested.

        Parameters
        ----------
        x
            Batch of instances.

        Returns
        -------
        p-value obtained from the permutation test, the LSDD between the reference and test set, \
        and the LSDD threshold above which drift is flagged.
        """
        x_ref, x = self.preprocess(x)
        x_ref = torch.from_numpy(x_ref).to(self.device)  # type: ignore[assignment]
        x = torch.from_numpy(x).to(self.device)  # type: ignore[assignment]

        if self.preprocess_fn is not None and self.preprocess_at_init is False and not self.x_ref_preprocessed:
            self._configure_normalization(x_ref)  # type: ignore[arg-type]
            x_ref = self._normalize(x_ref)
            self._initialize_kernel(x_ref)  # type: ignore[arg-type]
            self._configure_kernel_centers(x_ref)  # type: ignore[arg-type]
            self.H = GaussianRBF(np.sqrt(2.) * self.kernel.sigma)(self.kernel_centers, self.kernel_centers)

        x = self._normalize(x)

        k_yc = self.kernel(x, self.kernel_centers)
        k_all_c = torch.cat([self.k_xc, k_yc], 0)

        n_x = x_ref.shape[0] - self.n_kernel_centers
        n_all = k_all_c.shape[0]
        perms = [torch.randperm(n_all) for _ in range(self.n_permutations)]
        x_perms = [perm[:n_x] for perm in perms]
        y_perms = [perm[n_x:] for perm in perms]

        lsdd_permuted, _, lsdd = permed_lsdds(  # type: ignore
            k_all_c, x_perms, y_perms, self.H, lam_rd_max=self.lambda_rd_max, return_unpermed=True
        )
        p_val = (lsdd <= lsdd_permuted).float().mean()

        idx_threshold = int(self.p_val * len(lsdd_permuted))
        distance_threshold = torch.sort(lsdd_permuted, descending=True).values[idx_threshold]
        return float(p_val.cpu()), float(lsdd.cpu().numpy()), distance_threshold.cpu().numpy()
