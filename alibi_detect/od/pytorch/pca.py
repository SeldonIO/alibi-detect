from typing import Optional, Union, Callable

import torch

from alibi_detect.od.pytorch.base import TorchOutlierDetector


class PCATorch(TorchOutlierDetector):
    def __init__(
            self,
            n_components: int,
            device: Optional[Union[str, torch.device]] = None
            ):
        """PyTorch backend for PCA detector.

        Parameters
        ----------
        n_components:
            The number of dimensions in the principle subspace. For linear pca should have
            1 <= n_components < dim(data). For kernel pca should have 1 <= n_components < len(data).
        device
            Device type used. The default None tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either ``'cuda'``, ``'gpu'`` or ``'cpu'``.
        """
        TorchOutlierDetector.__init__(self, device=device)
        self.accumulator = None
        self.n_components = n_components

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Detect if `x` is an outlier.

        Parameters
        ----------
        x
            `torch.Tensor` with leading batch dimension.

        Returns
        -------
        `torch.Tensor` of ``bool`` values with leading batch dimension.

        Raises
        ------
        ThresholdNotInferredException
            If called before detector has had `infer_threshold` method called.
        """
        raw_scores = self.score(x)
        scores = self._accumulator(raw_scores)
        if not torch.jit.is_scripting():
            self.check_threshold_infered()
        preds = scores > self.threshold
        return preds.cpu()

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Score test instance `x`.

        Parameters
        ----------
        x
            The tensor of instances. First dimension corresponds to batch.

        Returns
        -------
            Tensor of scores for each element in `x`.

        Raises
        ------
        NotFitException
            If called before detector has been fit.
        """
        if not torch.jit.is_scripting():
            self.check_fitted()
        score = self._compute_score(x)
        return score.cpu()

    def _fit(self, x_ref: torch.Tensor) -> None:
        """Fits the pca detector.

        Parameters
        ----------
        x_ref
            The Dataset tensor.
        """
        self.x_ref_mean = x_ref.mean(0)
        self.pcs = self._compute_pcs(x_ref)
        self.x_ref = x_ref

    def _compute_pcs(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _compute_score(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LinearPCATorch(PCATorch):
    def __init__(
            self,
            n_components: int,
            device: Optional[Union[str, torch.device]] = None
            ):
        """Linear variant of the PyTorch backend for PCA detector.

        Parameters
        ----------
        n_components:
            The number of dimensions in the principle subspace. For linear pca should have
            1 <= n_components < dim(data). For kernel pca should have 1 <= n_components < len(data).
        device
            Device type used. The default None tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either ``'cuda'``, ``'gpu'`` or ``'cpu'``.
        """
        PCATorch.__init__(self, device=device, n_components=n_components)

    def _compute_pcs(self, x: torch.Tensor) -> torch.Tensor:
        x -= self.x_ref_mean
        cov_mat = (x.t() @ x)/(len(x)-1)
        _, V = torch.linalg.eigh(cov_mat)
        return V[:, :-self.n_components]

    def _compute_score(self, x: torch.Tensor) -> torch.Tensor:
        x_cen = x - self.x_ref_mean
        x_pcs = x_cen @ self.pcs
        return (x_pcs**2).sum(1)


class KernelPCATorch(PCATorch):
    def __init__(
            self,
            n_components: int,
            kernel: Optional[Callable],
            device: Optional[Union[str, torch.device]] = None
            ):
        """Kernel variant of the PyTorch backend for PCA detector.

        Parameters
        ----------
        n_components:
            The number of dimensions in the principle subspace. For linear pca should have
            1 <= n_components < dim(data). For kernel pca should have 1 <= n_components < len(data).
        kernel
            Kernel function to use for outlier detection. If ``None``, linear PCA is used instead of the
            kernel variant.
        device
            Device type used. The default None tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either ``'cuda'``, ``'gpu'`` or ``'cpu'``.
        """
        PCATorch.__init__(self, device=device, n_components=n_components)
        self.kernel = kernel

    def _compute_pcs(self, x: torch.Tensor) -> torch.Tensor:
        K = self._compute_kernel_cov_mat(x)
        D, V = torch.linalg.eigh(K)
        pcs = V / torch.sqrt(D)[None, :]
        return pcs[:, -self.n_components:]

    def _compute_score(self, x: torch.Tensor) -> torch.Tensor:
        k_xr = self.kernel(x, self.x_ref)
        # Now to center
        k_xr_row_sums = k_xr.sum(1)
        _, n = k_xr.shape
        k_xr_cen = k_xr - self.k_col_sums[None, :]/n - k_xr_row_sums[:, None]/n + self.k_sum/(n**2)
        x_pcs = k_xr_cen @ self.pcs
        scores = -2 * k_xr.mean(-1) - (x_pcs**2).sum(1)
        return scores

    def _compute_kernel_cov_mat(self, x: torch.Tensor) -> torch.Tensor:
        n = len(x)
        # Uncentered kernel matrix
        k = self.kernel(x, x)
        # Now to center
        self.k_col_sums = k.sum(0)
        k_row_sums = k.sum(1)
        self.k_sum = k_row_sums.sum()
        k_cen = k - self.k_col_sums[None, :]/n - k_row_sums[:, None]/n + self.k_sum/(n**2)
        return k_cen
