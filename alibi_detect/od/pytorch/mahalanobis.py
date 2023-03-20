from typing import Optional, Union

import torch

from alibi_detect.od.pytorch.base import TorchOutlierDetector


class MahalanobisTorch(TorchOutlierDetector):
    ensemble = None

    def __init__(
            self,
            min_eigenvalue: float = 1e-6,
            device: Optional[Union[str, torch.device]] = None
            ):
        """PyTorch backend for Mahalanobis detector.

        Parameters
        ----------
        min_eigenvalue
            Eigenvectors with eigenvalues below this value will be discarded.
        device
            Device type used. The default None tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either ``'cuda'``, ``'gpu'`` or ``'cpu'``.
        """
        TorchOutlierDetector.__init__(self, device=device)
        self.min_eigenvalue = min_eigenvalue

    @torch.no_grad()
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
        return preds.cpu()

    @torch.no_grad()
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
        x = torch.as_tensor(x)
        x_pcs = self._compute_linear_proj(x)
        return (x_pcs**2).sum(-1).cpu()

    def fit(self, x_ref: torch.Tensor):
        """Fits the detector

        Parameters
        ----------
        x_ref
            The Dataset tensor.
        """
        self.x_ref = x_ref
        self._compute_linear_pcs(self.x_ref)
        self.set_fitted()

    def _compute_linear_pcs(self, X: torch.Tensor):
        """Computes the principle components of the data.

        Parameters
        ----------
        X
            The reference dataset.
        """
        self.means = X.mean(0)
        X = X - self.means
        cov_mat = (X.t() @ X)/(len(X)-1)
        D, V = torch.linalg.eigh(cov_mat)
        non_zero_inds = D > self.min_eigenvalue
        self.pcs = V[:, non_zero_inds] / D[None,  non_zero_inds].sqrt()

    def _compute_linear_proj(self, X: torch.Tensor) -> torch.Tensor:
        """Projects the data point being tested onto the principle components.

        Parameters
        ----------
        X
            The data point being tested.
        """
        X_cen = X - self.means
        X_proj = X_cen @ self.pcs
        return X_proj
