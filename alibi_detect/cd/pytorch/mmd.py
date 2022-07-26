import logging
import numpy as np
import scipy.stats as stats
import torch
from typing import Callable, Dict, Optional, Tuple, Union
from alibi_detect.cd.base import BaseMMDDrift
from alibi_detect.utils.pytorch.distance import mmd2_from_kernel_matrix, linear_mmd2
from alibi_detect.utils.pytorch import get_device
from alibi_detect.utils.pytorch.kernels import GaussianRBF
from alibi_detect.utils.warnings import deprecated_alias

logger = logging.getLogger(__name__)


class MMDDriftTorch(BaseMMDDrift):
    @deprecated_alias(preprocess_x_ref='preprocess_at_init')
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
        self.meta.update({'backend': 'pytorch'})

        # set device
        self.device = get_device(device)

        # initialize kernel
        sigma = torch.from_numpy(sigma).to(self.device) if isinstance(sigma,  # type: ignore[assignment]
                                                                      np.ndarray) else None
        self.kernel = kernel(sigma) if kernel == GaussianRBF else kernel

        # compute kernel matrix for the reference data
        if self.infer_sigma or isinstance(sigma, torch.Tensor):
            x = torch.from_numpy(self.x_ref).to(self.device)
            self.k_xx = self.kernel(x, x, infer_sigma=self.infer_sigma)
            self.infer_sigma = False
        else:
            self.k_xx, self.infer_sigma = None, True

    def kernel_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ Compute and return full kernel matrix between arrays x and y. """
        k_xy = self.kernel(x, y, self.infer_sigma)
        k_xx = self.k_xx if self.k_xx is not None and self.update_x_ref is None else self.kernel(x, x)
        k_yy = self.kernel(y, y)
        kernel_mat = torch.cat([torch.cat([k_xx, k_xy], 1), torch.cat([k_xy.T, k_yy], 1)], 0)
        return kernel_mat

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
        p-value obtained from the permutation test, the MMD^2 between the reference and test set,
        and the MMD^2 threshold above which drift is flagged.
        """
        x_ref, x = self.preprocess(x)
        n = x.shape[0]
        x_ref = torch.from_numpy(x_ref).to(self.device)  # type: ignore[assignment]
        x = torch.from_numpy(x).to(self.device)  # type: ignore[assignment]
        # compute kernel matrix, MMD^2 and apply permutation test using the kernel matrix
        # TODO: (See https://github.com/SeldonIO/alibi-detect/issues/540)
        n = x.shape[0]  # type: ignore
        kernel_mat = self.kernel_matrix(x_ref, x)  # type: ignore[arg-type]
        kernel_mat = kernel_mat - torch.diag(kernel_mat.diag())  # zero diagonal
        mmd2 = mmd2_from_kernel_matrix(kernel_mat, n, permute=False, zero_diag=False)  # type: ignore[assignment]
        mmd2_permuted = torch.Tensor(
            [mmd2_from_kernel_matrix(kernel_mat, n, permute=True, zero_diag=False)
             for _ in range(self.n_permutations)]
            )
        if self.device.type == 'cuda':
            mmd2, mmd2_permuted = mmd2.cpu(), mmd2_permuted.cpu()
        p_val = (mmd2 <= mmd2_permuted).float().mean()
        # compute distance threshold
        idx_threshold = int(self.p_val * len(mmd2_permuted))
        distance_threshold = torch.sort(mmd2_permuted, descending=True).values[idx_threshold]
        return p_val.numpy().item(), mmd2.numpy().item(), distance_threshold.numpy()


class LinearTimeMMDDriftTorch(BaseMMDDrift):
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
            device: Optional[str] = None,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Maximum Mean Discrepancy (MMD) data drift detector using a linear-time estimator.

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
            p_val=p_val,
            x_ref_preprocessed=x_ref_preprocessed,
            preprocess_at_init=preprocess_at_init,
            update_x_ref=update_x_ref,
            preprocess_fn=preprocess_fn,
            sigma=sigma,
            configure_kernel_from_x_ref=configure_kernel_from_x_ref,
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
        sigma = torch.from_numpy(sigma).to(self.device) if isinstance(sigma,  # type: ignore[assignment]
                                                                      np.ndarray) else None
        self.kernel = kernel(sigma) if kernel == GaussianRBF else kernel

        # compute kernel matrix for the reference data
        if self.infer_sigma or isinstance(sigma, torch.Tensor):
            n = self.x_ref.shape[0]
            n_hat = int(np.floor(n / 2) * 2)
            x = torch.from_numpy(self.x_ref[:n_hat, :]).to(self.device)
            self.k_xx = self.kernel(x=x[0::2, :], y=x[1::2, :],
                                    pairwise=False, infer_sigma=self.infer_sigma)
            self.infer_sigma = False
        else:
            self.k_xx, self.infer_sigma = None, True

    def kernel_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ Compute and return full kernel matrix between arrays x and y. """
        k_xy = self.kernel(x, y, self.infer_sigma)
        k_xx = self.k_xx if self.k_xx is not None and self.update_x_ref is None else self.kernel(x, x)
        k_yy = self.kernel(y, y)
        kernel_mat = torch.cat([torch.cat([k_xx, k_xy], 1), torch.cat([k_xy.T, k_yy], 1)], 0)
        return kernel_mat

    def score(self, x: Union[np.ndarray, list]) -> Tuple[float, float, float]:
        """
        Compute the p-value using the maximum mean discrepancy as a distance measure between the
        reference data and the data to be tested. x and x_ref are required to have the same size.
        The sample size is then specified as the maximal even number below the data size.

        Parameters
        ----------
        x
            Batch of instances.

        Returns
        -------
        p-value obtained from the null hypothesis, the MMD^2 between the reference and test set
        and the MMD^2 threshold for the given significance level.
        """
        x_ref, x = self.preprocess(x)
        n = x.shape[0]
        m = x_ref.shape[0]
        if n != m:
            raise ValueError('x and x_ref must have the same size.')
        n_hat = int(np.floor(n / 2) * 2)
        x_ref = torch.from_numpy(x_ref[:n_hat, :]).to(self.device)  # type: ignore[assignment]
        x = torch.from_numpy(x[:n_hat, :]).to(self.device)  # type: ignore[assignment]
        if self.k_xx is not None and self.update_x_ref is None:
            k_xx = self.k_xx
        else:
            k_xx = self.kernel(x=x_ref[0::2, :], y=x_ref[1::2, :], pairwise=False)
        mmd2, var_mmd2 = linear_mmd2(k_xx, x_ref, x, self.kernel)  # type: ignore[arg-type]
        if self.device.type == 'cuda':
            mmd2 = mmd2.cpu()
        mmd2 = mmd2.numpy().item()
        var_mmd2 = np.clip(var_mmd2.numpy().item(), 1e-8, 1e8)
        std_mmd2 = np.sqrt(var_mmd2)
        t = mmd2 / (std_mmd2 / np.sqrt(n_hat / 2.))
        p_val = 1 - stats.t.cdf(t, df=(n_hat / 2.) - 1)
        distance_threshold = stats.t.ppf(1 - self.p_val, df=(n_hat / 2.) - 1)
        return p_val, t, distance_threshold
