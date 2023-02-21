from typing import Optional, Union, List, Tuple

import numpy as np
import torch

from alibi_detect.od.pytorch.ensemble import Ensembler
from alibi_detect.od.pytorch.base import TorchOutlierDetector


class LOFTorch(TorchOutlierDetector):
    def __init__(
            self,
            k: Union[np.ndarray, List, Tuple],
            kernel: Optional[torch.nn.Module] = None,
            ensembler: Optional[Ensembler] = None,
            device: Optional[Union[str, torch.device]] = None
            ):
        """PyTorch backend for LOF detector.

        Computes the Local Outlier Factor (LOF) of each instance in `x` with respect to a reference set `x_ref`.

        Parameters
        ----------
        k
            Number of nearest neighbors used to compute LOF. If `k` is a list or array, then an ensemble of LOF
            detectors is created with one detector for each value of `k`.
        kernel
            If a kernel is specified then instead of using `torch.cdist` the kernel defines the `k` nearest
            neighbor distance.
        ensembler
            If `k` is an array of integers then the ensembler must not be ``None``. Should be an instance
            of :py:obj:`alibi_detect.od.pytorch.ensemble.ensembler`. Responsible for combining
            multiple scores into a single score.
        device
            Device on which to run the detector.
        """
        TorchOutlierDetector.__init__(self, device=device)
        self.kernel = kernel
        self.ensemble = isinstance(k, (np.ndarray, list, tuple))
        self.ks = torch.tensor(k) if self.ensemble else torch.tensor([k], device=self.device)
        self.ensembler = ensembler

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
        raw_scores = self.score(x)
        scores = self._ensembler(raw_scores)
        if not torch.jit.is_scripting():
            self.check_threshold_inferred()
        preds = scores > self.threshold
        return preds

    def _make_mask(self, reachabilities: torch.Tensor):
        mask = torch.zeros_like(reachabilities[0], device=self.device)
        for i, k in enumerate(self.ks):
            mask[:k, i] = torch.ones(k, device=self.device)/k
        return mask

    def _compute_K(self, x, y):
        return torch.exp(-self.kernel(x, y)) if self.kernel is not None else torch.cdist(x, y)

    @torch.no_grad()
    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the score of `x`

        The score step proceeds as follows:
        1. Compute the distance between each instance in `x` and the reference set.
        2. Compute the k-nearest neighbors of each instance in `x` in the reference set.
        3. Compute the reachability distance of each instance in `x` to its k-nearest neighbors.
        4. For each instance sum the inv_avg_reachabilities of its neighbours.
        5. LOF is average reachability of instance over average reachability of neighbours.


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

        X = torch.as_tensor(x)
        D = self._compute_K(X, self.x_ref)
        max_k = torch.max(self.ks)
        bot_k_items = torch.topk(D, int(max_k), dim=1, largest=False)
        bot_k_inds, bot_k_dists = bot_k_items.indices, bot_k_items.values
        lower_bounds = self.knn_dists_ref[bot_k_inds]
        reachabilities = torch.max(bot_k_dists[:, :, None], lower_bounds)
        mask = self._make_mask(reachabilities)
        avg_reachabilities = (reachabilities*mask[None, :, :]).sum(1)
        factors = (self.ref_inv_avg_reachabilities[bot_k_inds]*mask[None, :, :]).sum(1)
        lofs = (avg_reachabilities * factors)
        return lofs if self.ensemble else lofs[:, 0]

    @torch.no_grad()
    def _fit(self, x_ref: torch.Tensor):
        """Fits the detector

        The LOF algorithm fit step prodeeds as follows:
        1. Compute the distance matrix, D, between all instances in `x_ref`.
        2. For each instance, compute the k nearest neighbours. (Note we prevent an instance from
            considering itself a neighbour by setting the diagonal of D to be the maximum value of D.)
        3. For each instance we store the distance to its kth nearest neighbour for each k in `ks`.
        4. For each instance and k in `ks` we obtain a tensor of the k neighbours k nearest neighbour
            distances.
        5. The reachability of an instance is the maximum of its k nearest neighbours distances and
            the distance to its kth nearest neighbour.
        6. The reachabilites tensor is of shape `(n_instances, max(ks), len(ks))`. Where the second
            dimension is the each of the k neighbours nearest distances and the third dimension is
            the specific k.
        7. The local reachability density is then given by 1 over the average reachability
            over the second dimension of this tensor. However we only want to consider the k nearest
            neighbours for each k in `ks`, so we use a mask that prevents k from the second dimension
            greater than k from the third dimension from being considered. This value is stored as
            we use it in the score step.
        8. If multiple k are passed in ks then the detector also needs to fit the ensembler. To do so
            we need to score the x_ref as well. The local outlier factor (LOF) is then given by the
            average reachability of an instance over the average reachability of its k neighbours.

        Parameters
        ----------
        x_ref
            The Dataset tensor.
        """
        X = torch.as_tensor(x_ref)
        D = self._compute_K(X, X)
        D += torch.eye(len(D), device=self.device) * torch.max(D)
        max_k = torch.max(self.ks)
        bot_k_items = torch.topk(D, int(max_k), dim=1, largest=False)
        bot_k_inds, bot_k_dists = bot_k_items.indices, bot_k_items.values
        self.knn_dists_ref = bot_k_dists[:, self.ks-1]
        lower_bounds = self.knn_dists_ref[bot_k_inds]
        reachabilities = torch.max(bot_k_dists[:, :, None], lower_bounds)
        mask = self._make_mask(reachabilities)
        avg_reachabilities = (reachabilities*mask[None, :, :]).sum(1)
        self.ref_inv_avg_reachabilities = 1/avg_reachabilities
        self.x_ref = X

        if self.ensemble:
            factors = (self.ref_inv_avg_reachabilities[bot_k_inds]*mask[None, :, :]).sum(1)
            scores = (avg_reachabilities * factors)
            self.ensembler.fit(scores)
