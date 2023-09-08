from typing import Optional, Union, List, Tuple
import numpy as np
import torch

from alibi_detect.od.pytorch.ensemble import Ensembler
from alibi_detect.od.pytorch.base import TorchOutlierDetector
from alibi_detect.utils._types import TorchDeviceType


class KNNTorch(TorchOutlierDetector):
    def __init__(
            self,
            k: Union[np.ndarray, List, Tuple, int],
            kernel: Optional[torch.nn.Module] = None,
            ensembler: Optional[Ensembler] = None,
            device: TorchDeviceType = None,
            ):
        """PyTorch backend for KNN detector.

        Parameters
        ----------
        k
            Number of nearest neighbors to compute distance to. `k` can be a single value or
            an array of integers. If `k` is a single value the outlier score is the distance/kernel
            similarity to the `k`-th nearest neighbor. If `k` is a list then it returns the distance/kernel
            similarity to each of the specified `k` neighbors.
        kernel
            If a kernel is specified then instead of using `torch.cdist` the kernel defines the `k` nearest
            neighbor distance.
        ensembler
            If `k` is an array of integers then the ensembler must not be ``None``. Should be an instance
            of :py:obj:`alibi_detect.od.pytorch.ensemble.ensembler`. Responsible for combining
            multiple scores into a single score.
        device
            Device type used. The default tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of
            ``torch.device``.
        """
        super().__init__(device=device)
        self.kernel = kernel
        self.ensemble = isinstance(k, (np.ndarray, list, tuple))
        self.ks = torch.tensor(k) if self.ensemble else torch.tensor([k], device=self.device)
        self.ensembler = ensembler

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
        ThresholdNotInferredError
            If called before detector has had `infer_threshold` method called.
        """
        raw_scores = self.score(x)
        scores = self._ensembler(raw_scores)
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
        NotFittedError
            If called before detector has been fit.
        """
        self.check_fitted()
        K = -self.kernel(x, self.x_ref) if self.kernel is not None else torch.cdist(x, self.x_ref)
        bot_k_dists = torch.topk(K, int(torch.max(self.ks)), dim=1, largest=False)
        all_knn_dists = bot_k_dists.values[:, self.ks-1]
        return all_knn_dists if self.ensemble else all_knn_dists[:, 0]

    def fit(self, x_ref: torch.Tensor):
        """Fits the detector

        Parameters
        ----------
        x_ref
            The Dataset tensor.
        """
        self.x_ref = x_ref
        self._set_fitted()
