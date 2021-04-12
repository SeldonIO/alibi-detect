import numpy as np
import torch
from . import distance
from typing import Optional


class GaussianRBF:
    def __init__(self, sigma: Optional[torch.Tensor] = None) -> None:
        """
        Gaussian RBF kernel: k(x,y) = exp(-(1/(2*sigma^2)||x-y||^2). A forward pass takes
        a batch of instances x [Nx, features] and y [Ny, features] and returns the kernel
        matrix [Nx, Ny].

        Parameters
        ----------
        sigma
            Optional sigma used for the kernel.
        """
        super().__init__()
        self.sigma = sigma

    def __call__(self, x: torch.Tensor, y: torch.Tensor, infer_sigma: bool = False) -> torch.Tensor:

        dist = distance.squared_pairwise_distance(x, y)  # [Nx, Ny]

        if infer_sigma:
            n = min(x.shape[0], y.shape[0])
            n = n if (x[:n] == y[:n]).all() and x.shape == y.shape else 0
            n_median = n + (np.prod(dist.shape) - n) // 2 - 1
            self.sigma = (.5 * dist.flatten().sort().values[n_median].unsqueeze(dim=-1)) ** .5

        gamma = 1. / (2. * self.sigma ** 2)   # [Ns,]
        # TODO: do matrix multiplication after all?
        kernel_mat = torch.exp(- torch.cat([(g * dist)[None, :, :] for g in gamma], dim=0))  # [Ns, Nx, Ny]
        return kernel_mat.mean(dim=0)  # [Nx, Ny]
