import numpy as np
from pykeops.torch import LazyTensor
import torch
import torch.nn as nn
from typing import Callable, Optional


def sigma_mean(x: LazyTensor, y: LazyTensor, dist: LazyTensor) -> torch.Tensor:
    """
    Bandwidth estimation using the mean heuristic.

    Parameters
    ----------
    x
        LazyTensor of instances with dimension [Nx, 1, features].
    y
        LazyTensor of instances with dimension [1, Ny, features].
    dist
        LazyTensor with dimensions [Nx, Ny], containing the pairwise distances between `x` and `y`.

    Returns
    -------
    The computed bandwidth, `sigma`.
    """
    n = x.shape[0]
    if (dist.min(axis=1) == 0.).all() and (torch.arange(n) == dist.argmin(axis=1).cpu().view(-1)).all() \
            and x.shape == y.shape:
        n_mean = n * (n - 1)
    else:
        n_mean = np.prod(dist.shape)
    sigma = (.5 * dist.sum(1).sum().unsqueeze(-1) / n_mean) ** .5
    return sigma


class GaussianRBF(nn.Module):
    def __init__(
        self,
        sigma: Optional[torch.Tensor] = None,
        init_sigma_fn: Callable = sigma_mean,
        trainable: bool = False
    ) -> None:
        """
        Gaussian RBF kernel: k(x,y) = exp(-(1/(2*sigma^2)||x-y||^2). A forward pass takes
        a batch of instances x [Nx, features] and y [Ny, features] and returns the kernel
        matrix [Nx, Ny].

        Parameters
        ----------
        sigma
            Bandwidth used for the kernel. Needn't be specified if being inferred or trained.
            Can pass multiple values to eval kernel with and then average.
        init_sigma_fn
            Function used to compute the bandwidth `sigma`. Used when `sigma` is to be inferred.
            The function's signature should match :py:func:`~alibi_detect.utils.keops.kernels.sigma_mean`,
            meaning that it should take in the tensors `x`, `y` and `dist` and return `sigma`.
        trainable
            Whether or not to track gradients w.r.t. `sigma` to allow it to be trained.
        """
        super().__init__()
        if sigma is None:
            self.log_sigma = nn.Parameter(torch.empty(1), requires_grad=trainable)
            self.init_required = True
        else:
            sigma = sigma.reshape(-1)  # [Ns,]
            self.log_sigma = nn.Parameter(sigma.log(), requires_grad=trainable)
            self.init_required = False
        self.init_sigma_fn = init_sigma_fn
        self.trainable = trainable

    @property
    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp()

    def forward(self, x: LazyTensor, y: LazyTensor, infer_sigma: bool = False) -> LazyTensor:

        dist = ((x - y) ** 2).sum(-1)

        if infer_sigma or self.init_required:
            if self.trainable and infer_sigma:
                raise ValueError("Gradients cannot be computed w.r.t. an inferred sigma value")
            sigma = self.init_sigma_fn(x, y, dist)  # .to(x.device)
            with torch.no_grad():
                self.log_sigma.copy_(sigma.log().clone())
            self.init_required = False

        gamma = 1. / (2. * self.sigma ** 2)
        gamma = LazyTensor(gamma[None, None, :]) if len(dist.shape) == 2 else LazyTensor(gamma[None, None, None, :])
        kernel_mat = (- gamma * dist).exp()
        if len(dist.shape) < len(gamma.shape):
            kernel_mat = kernel_mat.sum(-1) / len(self.sigma)
        return kernel_mat
