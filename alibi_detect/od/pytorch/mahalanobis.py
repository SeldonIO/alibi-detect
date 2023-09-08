import torch

from alibi_detect.od.pytorch.base import TorchOutlierDetector
from alibi_detect.utils._types import TorchDeviceType


class MahalanobisTorch(TorchOutlierDetector):
    ensemble = False

    def __init__(
            self,
            min_eigenvalue: float = 1e-6,
            device: TorchDeviceType = None,
            ):
        """PyTorch backend for Mahalanobis detector.

        Parameters
        ----------
        min_eigenvalue
            Eigenvectors with eigenvalues below this value will be discarded.
        device
            Device type used. The default tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of
            ``torch.device``.
        """
        super().__init__(device=device)
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
        x_pcs = self._compute_linear_proj(x)
        return (x_pcs**2).sum(-1)

    def fit(self, x_ref: torch.Tensor):
        """Fits the detector

        Parameters
        ----------
        x_ref
            The Dataset tensor.
        """
        self.x_ref = x_ref
        self._compute_linear_pcs(self.x_ref)
        self._set_fitted()

    def _compute_linear_pcs(self, x: torch.Tensor):
        """Computes the principal components of the data.

        Parameters
        ----------
        x
            The reference dataset.
        """
        self.means = x.mean(0)
        x = x - self.means
        cov_mat = (x.t() @ x)/(len(x)-1)
        D, V = torch.linalg.eigh(cov_mat)
        non_zero_inds = D > self.min_eigenvalue
        self.pcs = V[:, non_zero_inds] / D[None,  non_zero_inds].sqrt()

    def _compute_linear_proj(self, x: torch.Tensor) -> torch.Tensor:
        """Projects the data point being tested onto the principal components.

        Parameters
        ----------
        x
            The data point being tested.
        """
        x_cen = x - self.means
        x_proj = x_cen @ self.pcs
        return x_proj
