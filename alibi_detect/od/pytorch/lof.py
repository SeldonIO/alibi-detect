from typing import Optional, Union, List, Tuple
import numpy as np
import torch

from alibi_detect.od.pytorch.ensemble import Ensembler
from alibi_detect.od.pytorch.base import TorchOutlierDetector
from alibi_detect.utils._types import TorchDeviceType


class LOFTorch(TorchOutlierDetector):
    def __init__(
            self,
            k: Union[np.ndarray, List, Tuple, int],
            kernel: Optional[torch.nn.Module] = None,
            ensembler: Optional[Ensembler] = None,
            device: TorchDeviceType = None,
            ):
        """PyTorch backend for LOF detector.

        Parameters
        ----------
        k
            Number of nearest neighbors used to compute the local outlier factor. `k` can be a single
            value or an array of integers. If `k` is a single value the score method uses the
            distance/kernel similarity to the `k`-th nearest neighbor. If `k` is a list then it uses
            the distance/kernel similarity to each of the specified `k` neighbors.
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
        TorchOutlierDetector.__init__(self, device=device)
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

    def _make_mask(self, reachabilities: torch.Tensor):
        """Generate a mask for computing the average reachability.

        If k is an array then we need to compute the average reachability for each k separately. To do
        this we use a mask to weight the reachability of each k-close neighbor by 1/k and the rest to 0.
        """
        mask = torch.zeros_like(reachabilities[0], device=self.device)
        for i, k in enumerate(self.ks):
            mask[:k, i] = torch.ones(k, device=self.device)/k
        return mask

    def _compute_K(self, x, y):
        """Compute the distance matrix matrix between `x` and `y`."""
        return torch.exp(-self.kernel(x, y)) if self.kernel is not None else torch.cdist(x, y)

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

        # compute the distance matrix between x and x_ref
        K = self._compute_K(x, self.x_ref)

        # compute k nearest neighbors for maximum k in self.ks
        max_k = torch.max(self.ks)
        bot_k_items = torch.topk(K, int(max_k), dim=1, largest=False)
        bot_k_inds, bot_k_dists = bot_k_items.indices, bot_k_items.values

        # To compute the reachabilities we get the k-distances of each object in the instances
        # k nearest neighbors. Then we take the maximum of their k-distances and the distance
        # to the instance.
        lower_bounds = self.knn_dists_ref[bot_k_inds]
        reachabilities = torch.max(bot_k_dists[:, :, None], lower_bounds)

        # Compute the average reachability for each instance. We use a mask to manage each k in
        # self.ks separately.
        mask = self._make_mask(reachabilities)
        avg_reachabilities = (reachabilities*mask[None, :, :]).sum(1)

        # Compute the LOF score for each instance. Note we don't take 1/avg_reachabilities as
        # avg_reachabilities is the denominator in the LOF formula.
        factors = (self.ref_inv_avg_reachabilities[bot_k_inds] * mask[None, :, :]).sum(1)
        lofs = (avg_reachabilities * factors)
        return lofs if self.ensemble else lofs[:, 0]

    def fit(self, x_ref: torch.Tensor):
        """Fits the detector

        Parameters
        ----------
        x_ref
            The Dataset tensor.
        """
        # compute the distance matrix
        K = self._compute_K(x_ref, x_ref)
        # set diagonal to max distance to prevent torch.topk from returning the instance itself
        K += torch.eye(len(K), device=self.device) * torch.max(K)

        # compute k nearest neighbors for maximum k in self.ks
        max_k = torch.max(self.ks)
        bot_k_items = torch.topk(K, int(max_k), dim=1, largest=False)
        bot_k_inds, bot_k_dists = bot_k_items.indices, bot_k_items.values

        # store the k-distances for each instance for each k.
        self.knn_dists_ref = bot_k_dists[:, self.ks-1]

        # To compute the reachabilities we get the k-distances of each object in the instances
        # k nearest neighbors. Then we take the maximum of their k-distances and the distance
        # to the instance.
        lower_bounds = self.knn_dists_ref[bot_k_inds]
        reachabilities = torch.max(bot_k_dists[:, :, None], lower_bounds)

        # Compute the average reachability for each instance. We use a mask to manage each k in
        # self.ks separately.
        mask = self._make_mask(reachabilities)
        avg_reachabilities = (reachabilities*mask[None, :, :]).sum(1)

        # Compute the inverse average reachability for each instance.
        self.ref_inv_avg_reachabilities = 1/avg_reachabilities

        self.x_ref = x_ref
        self._set_fitted()
