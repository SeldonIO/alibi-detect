import numpy as np
from pykeops.torch import LazyTensor
import torch
import torch.nn as nn
from typing import Callable, Optional


def sigma_mean(x: LazyTensor, y: LazyTensor, dist: LazyTensor, n_min: int = None) -> torch.Tensor:
    """
    Set bandwidth to the mean distance between instances x and y.

    Parameters
    ----------
    x
        LazyTensor of instances with dimension [Nx, 1, features]. The singleton dimension is necessary for broadcasting.
    y
        LazyTensor of instances with dimension [1, Ny, features]. The singleton dimension is necessary for broadcasting.
    dist
        LazyTensor with dimensions [Nx, Ny] containing the pairwise distances between `x` and `y`.
    n_min
        In order to check whether x equals y after squeezing the singleton dimensions, we check if the
        diagonal of the distance matrix (which is a lazy tensor from which the diagonal cannot be directly extracted)
        consists of all zeros. We do this by computing the k-min distances and k-argmin indices over the
        columns of the distance matrix. We then check if the distances on the diagonal of the distance matrix
        are all zero or not. If they are all zero, then we do not use these distances (zeros) when computing
        the mean pairwise distance as bandwidth. The default `None` sets k to Nx (=Ny). If Nx becomes very large,
        it is advised to set `n_min` to a lower value.

    Returns
    -------
    The computed bandwidth, `sigma`.
    """
    nx, ny = dist.shape
    if nx == ny:
        n_min = n_min if isinstance(n_min, int) else nx
        d_min, id_min = dist.Kmin_argKmin(n_min, axis=1)
        rows, cols = torch.where(id_min.cpu() == torch.arange(nx)[:, None])
        if (d_min[rows, cols] == 0.).all():
            n_mean = nx * (nx - 1)
        else:
            n_mean = np.prod(dist.shape)
    else:
        n_mean = np.prod(dist.shape)
    sigma = (.5 * dist.sum(1).sum().unsqueeze(-1) / n_mean) ** .5
    return sigma


class GaussianRBF(nn.Module):
    def __init__(
        self,
        sigma: Optional[torch.Tensor] = None,
        init_sigma_fn: Callable = None,
        trainable: bool = False
    ) -> None:
        """
        Gaussian RBF kernel: k(x,y) = exp(-(1/(2*sigma^2)||x-y||^2). A forward pass takes
        a batch of instances x and y and returns the kernel matrix.
        x can be of shape [Nx, 1, features] or [batch_size, Nx, 1, features].
        y can be of shape [1, Ny, features] or [batch_size, 1, Ny, features].
        The returned kernel matrix can be of shape [Nx, Ny] or [batch_size, Nx, Ny].
        x, y and the returned kernel matrix are all lazy tensors.

        Parameters
        ----------
        sigma
            Bandwidth used for the kernel. Needn't be specified if being inferred or trained.
            Can pass multiple values to eval kernel with and then average.
        init_sigma_fn
            Function used to compute the bandwidth `sigma`. Used when `sigma` is to be inferred.
            The function's signature should match :py:func:`~alibi_detect.utils.keops.kernels.sigma_mean`,
            meaning that it should take in the lazy tensors `x`, `y` and `dist` and return a tensor `sigma`.
        trainable
            Whether or not to track gradients w.r.t. `sigma` to allow it to be trained.
        """
        super().__init__()
        init_sigma_fn = sigma_mean if init_sigma_fn is None else init_sigma_fn
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
            sigma = self.init_sigma_fn(x, y, dist)
            with torch.no_grad():
                self.log_sigma.copy_(sigma.log().clone())
            self.init_required = False

        gamma = 1. / (2. * self.sigma ** 2)
        gamma = LazyTensor(gamma[None, None, :]) if len(dist.shape) == 2 else LazyTensor(gamma[None, None, None, :])
        kernel_mat = (- gamma * dist).exp()
        if len(dist.shape) < len(gamma.shape):
            kernel_mat = kernel_mat.sum(-1) / len(self.sigma)
        return kernel_mat
