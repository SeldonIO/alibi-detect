from typing import Optional, Union

import torch

from alibi_detect.od.pytorch.base import TorchOutlierDetector


class MahalanobisTorch(TorchOutlierDetector):
    def __init__(
            self,
            min_eigenvalue: float = 1e-6,
            device: Optional[Union[str, torch.device]] = None
            ):
        """PyTorch backend for KNN detector.

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
            self.check_threshold_infered()
        preds = scores > self.threshold
        return preds.cpu()

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the score of `x`

        Project onto the PCs.

        Note: that if one computes ``x_ref_proj = self._compute_method_proj(self.x_ref)``
        then one can check that each column has zero mean and unit variance. The idea
        is that new data will be similarly distributed if from the same distribution and therefore
        its distance from the origin forms a sensible outlier score.

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
        x = torch.as_tensor(x)
        x_pcs = self._compute_linear_proj(x)
        return (x_pcs**2).sum(-1).cpu()

    def _fit(self, x_ref: torch.Tensor):
        """Fits the detector

        Parameters
        ----------
        x_ref
            The Dataset tensor.
        """
        self.x_ref = x_ref
        self._compute_linear_pcs(self.x_ref)
        # As a sanity check one can call x_ref_proj = self._compute_method_proj(self.x_ref) and see that
        # we have fully whitened the data: each column has mean 0 and std 1.

    def _compute_linear_pcs(self, X: torch.Tensor):
        """
        This saves the *residual* pcs (those whose eigenvalues are not in
        the largest n_components). These are all that are needed to compute
        the reconstruction error in the linear case.
        """
        self.means = X.mean(0)
        X = X - self.means
        cov_mat = (X.t() @ X)/(len(X)-1)
        D, V = torch.linalg.eigh(cov_mat)
        non_zero_inds = D > self.min_eigenvalue
        self.pcs = V[:, non_zero_inds] / D[None,  non_zero_inds].sqrt()

    def _compute_linear_proj(self, X: torch.Tensor) -> torch.Tensor:
        X_cen = X - self.means
        X_proj = X_cen @ self.pcs
        return X_proj
