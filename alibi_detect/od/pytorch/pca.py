from typing import Optional, Callable

import torch

from alibi_detect.od.pytorch.base import TorchOutlierDetector
from alibi_detect.utils._types import TorchDeviceType


class PCATorch(TorchOutlierDetector):
    ensemble = False

    def __init__(
            self,
            n_components: int,
            device: TorchDeviceType = None,
            ):
        """PyTorch backend for PCA detector.

        Parameters
        ----------
        n_components:
            The number of dimensions in the principal subspace. For linear PCA should have
            ``1 <= n_components < dim(data)``. For kernel pca should have ``1 <= n_components < len(data)``.
        device
            Device type used. The default tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of
            ``torch.device``.

        Raises
        ------
        ValueError
            If `n_components` is less than 1.
        """
        super().__init__(device=device)
        self.n_components = n_components

        if n_components < 1:
            raise ValueError('n_components must be at least 1')

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
        scores = self.score(x)
        if not torch.jit.is_scripting():
            self.check_threshold_inferred()
        preds = scores > self.threshold
        return preds

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the score of `x`

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
        self.check_fitted()
        score = self._score(x)
        return score

    def fit(self, x_ref: torch.Tensor) -> None:
        """Fits the PCA detector.

        Parameters
        ----------
        x_ref
            The Dataset tensor.
        """
        self.pcs = self._fit(x_ref)
        self._set_fitted()

    def _fit(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _score(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LinearPCATorch(PCATorch):
    def __init__(
            self,
            n_components: int,
            device: TorchDeviceType = None,
            ):
        """Linear variant of the PyTorch backend for PCA detector.

        Parameters
        ----------
        n_components:
            The number of dimensions in the principal subspace.
        device
            Device type used. The default tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of
            ``torch.device``.
        """
        super().__init__(device=device, n_components=n_components)

    def _fit(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the principal components of the reference data.

        We compute the principal components of the reference data using the covariance matrix and then
        remove the largest `n_components` eigenvectors. The remaining eigenvectors correspond to the
        invariant dimensions of the data. Changes in these dimensions are used to compute the outlier
        score which is the distance to the principal subspace spanned by the first `n_components`
        eigenvectors.

        Parameters
        ----------
        x
            The reference data.

        Returns
        -------
        The principal components of the reference data.

        Raises
        ------
        ValueError
            If `n_components` is greater than or equal to number of features
        """
        if self.n_components >= x.shape[1]:
            raise ValueError("n_components must be less than the number of features.")

        self.x_ref_mean = x.mean(0)
        x -= self.x_ref_mean
        cov_mat = (x.t() @ x)/(len(x)-1)
        _, V = torch.linalg.eigh(cov_mat)
        return V[:, :-self.n_components]

    def _score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the outlier score.

        Centers the data and projects it onto the principal components. The score is then the sum of the
        squared projections.

        Parameters
        ----------
        x
            The test data.

        Returns
        -------
        The outlier score.
        """
        x_cen = x - self.x_ref_mean
        x_pcs = x_cen @ self.pcs
        return (x_pcs**2).sum(1)


class KernelPCATorch(PCATorch):
    def __init__(
            self,
            n_components: int,
            kernel: Optional[Callable],
            device: TorchDeviceType = None,
            ):
        """Kernel variant of the PyTorch backend for PCA detector.

        Parameters
        ----------
        n_components:
            The number of dimensions in the principal subspace.
        kernel
            Kernel function to use for outlier detection.
        device
            Device type used. The default tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of
            ``torch.device``.
        """
        super().__init__(device=device, n_components=n_components)
        self.kernel = kernel

    def _fit(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the principal components of the reference data.

        We compute the principal components of the reference data using the kernel matrix and then
        return the largest `n_components` eigenvectors. These are then normalized to have length
        equal to `1/eigenvalue`. Note that this differs from the linear case where we remove the
        largest eigenvectors.

        Parameters
        ----------
        x
            The reference data.

        Returns
        -------
        The principal components of the reference data.

        Raises
        ------
        ValueError
            If `n_components` is greater than or equal to the number of reference samples.
        """
        if self.n_components >= x.shape[0]:
            raise ValueError("n_components must be less than the number of reference instances.")

        self.x_ref = x
        K = self.compute_kernel_mat(x)
        D, V = torch.linalg.eigh(K)
        pcs = V / torch.sqrt(D)[None, :]
        return pcs[:, -self.n_components:]

    def _score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the outlier score.

        Centers the data and projects it onto the principal components. The score is then the sum of the
        squared projections.

        Parameters
        ----------
        x
            The test data.

        Returns
        -------
        The outlier score.
        """
        k_xr = self.kernel(x, self.x_ref)
        k_xr_row_sums = k_xr.sum(1)
        n, m = k_xr.shape
        k_xr_cen = k_xr - self.k_col_sums[None, :]/m - k_xr_row_sums[:, None]/n + self.k_sum/(m*n)
        x_pcs = k_xr_cen @ self.pcs
        scores = -2 * k_xr.mean(-1) - (x_pcs**2).sum(1)
        return scores

    def compute_kernel_mat(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the centered kernel matrix.

        Parameters
        ----------
        x
            The reference data.

        Returns
        -------
        The centered kernel matrix.
        """
        n = len(x)
        k = self.kernel(x, x)
        self.k_col_sums = k.sum(0)
        k_row_sums = k.sum(1)
        self.k_sum = k_row_sums.sum()
        k_cen = k - self.k_col_sums[None, :]/n - k_row_sums[:, None]/n + self.k_sum/(n**2)
        return k_cen
