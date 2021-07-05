import numpy as np
import torch
from torch import nn
from . import distance
from typing import Optional, Union


class GaussianRBF(nn.Module):
    def __init__(self, sigma: Optional[torch.tensor] = None, trainable: bool = False) -> None:
        """
        Gaussian RBF kernel: k(x,y) = exp(-(1/(2*sigma^2)||x-y||^2). A forward pass takes
        a batch of instances x [Nx, features] and y [Ny, features] and returns the kernel
        matrix [Nx, Ny].

        Parameters
        ----------
        sigma
            Bandwidth used for the kernel. Needn't be specified if being inferred or trained.
            Can pass multiple values to eval kernel with and then average.
        trainable
            Whether or not to track gradients w.r.t. sigma to allow it to be trained.
        """
        super().__init__()
        if trainable:
            sigma = sigma or torch.tensor(1.)
        if sigma is not None:
            sigma = sigma.reshape(-1)  # [Ns,]
            self.log_sigma = nn.Parameter(sigma.log(), requires_grad=trainable)
        self.trainable = trainable

    @property
    def sigma(self) -> torch.tensor:
        return self.log_sigma.exp()

    def forward(self, x: torch.Tensor, y: torch.Tensor, infer_sigma: bool = False) -> torch.Tensor:

        dist = distance.squared_pairwise_distance(x, y)  # [Nx, Ny]

        if infer_sigma:
            if self.trainable:
                raise ValueError("Gradients cannot be computed w.r.t. an inferred sigma value")
            n = min(x.shape[0], y.shape[0])
            n = n if (x[:n] == y[:n]).all() and x.shape == y.shape else 0
            n_median = n + (np.prod(dist.shape) - n) // 2 - 1
            sigma = (.5 * dist.flatten().sort().values[n_median].unsqueeze(dim=-1)) ** .5
            self.log_sigma = nn.Parameter(sigma.log(), requires_grad=False)

        gamma = 1. / (2. * self.sigma ** 2)   # [Ns,]
        # TODO: do matrix multiplication after all?
        kernel_mat = torch.exp(- torch.cat([(g * dist)[None, :, :] for g in gamma], dim=0))  # [Ns, Nx, Ny]
        return kernel_mat.mean(dim=0)  # [Nx, Ny]


class DeepKernel(nn.Module):
    """"
    Computes simmilarities as k(x,y) = (1-eps)*k_a(proj(x), proj(y)) + eps*k_b(x,y)
    """
    def __init__(
        self,
        proj: nn.Module,
        kernel_a: nn.Module = GaussianRBF(trainable=True),
        kernel_b: nn.Module = GaussianRBF(trainable=True),
        eps: Union[float, str] = 'trainable'
    ) -> None:
        super().__init__()

        self.proj = proj
        self.kernel_a = kernel_a
        self.kernel_b = kernel_b
        if isinstance(eps, float):
            if not 0 < eps < 1:
                raise ValueError("eps should be in (0,1)")
            self.logit_eps = nn.Parameter(torch.tensor(eps).logit(), requires_grad=False)
        elif eps == 'trainable':
            self.logit_eps = nn.Parameter(torch.tensor(0.))
        else:
            raise NotImplementedError("eps should be 'trainable' or a float in (0,1)")

    @property
    def eps(self) -> torch.tensor:
        return self.logit_eps.sigmoid()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return (1-self.eps)*self.kernel_a(self.proj(x), self.proj(y)) + self.eps*self.kernel_b(x, y)
