import numpy as np
import torch
from torch import nn
from . import distance
from typing import Optional, Union


class GaussianRBF(nn.Module):
    def __init__(
        self,
        sigma: Optional[torch.Tensor] = None,
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
        trainable
            Whether or not to track gradients w.r.t. sigma to allow it to be trained.
        """
        super().__init__()
        if sigma is None:
            self.log_sigma = nn.Parameter(torch.empty(1), requires_grad=trainable)
            self.init_required = True
        else:
            sigma = sigma.reshape(-1)  # [Ns,]
            self.log_sigma = nn.Parameter(sigma.log(), requires_grad=trainable)
            self.init_required = False
        self.trainable = trainable

    @property
    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp()

    def forward(self, x: torch.Tensor, y: torch.Tensor, infer_sigma: bool = False) -> torch.Tensor:

        dist = distance.squared_pairwise_distance(x.flatten(1), y.flatten(1))  # [Nx, Ny]

        if infer_sigma or self.init_required:
            if self.trainable and infer_sigma:
                raise ValueError("Gradients cannot be computed w.r.t. an inferred sigma value")
            n = min(x.shape[0], y.shape[0])
            n = n if (x[:n] == y[:n]).all() and x.shape == y.shape else 0
            n_median = n + (np.prod(dist.shape) - n) // 2 - 1
            sigma = (.5 * dist.flatten().sort().values[n_median].unsqueeze(dim=-1)) ** .5
            with torch.no_grad():
                self.log_sigma.copy_(sigma.log().clone())
            self.init_required = False

        gamma = 1. / (2. * self.sigma ** 2)   # [Ns,]
        # TODO: do matrix multiplication after all?
        kernel_mat = torch.exp(- torch.cat([(g * dist)[None, :, :] for g in gamma], dim=0))  # [Ns, Nx, Ny]
        return kernel_mat.mean(dim=0)  # [Nx, Ny]


class DeepKernel(nn.Module):
    """
    Computes similarities as k(x,y) = (1-eps)*k_a(proj(x), proj(y)) + eps*k_b(x,y).
    A forward pass takes a batch of instances x [Nx, features] and y [Ny, features] and returns
    the kernel matrix [Nx, Ny].

    Parameters
    ----------
    proj
        The projection to be applied to the inputs before applying kernel_a
    kernel_a
        The kernel to apply to the projected inputs. Defaults to a Gaussian RBF with trainable bandwidth.
    kernel_b
        The kernel to apply to the raw inputs. Defaults to a Gaussian RBF with trainable bandwidth.
        Set to None in order to use only the deep component (i.e. eps=0).
    eps
        The proportion (in [0,1]) of weight to assign to the kernel applied to raw inputs. This can be
        either specified or set to 'trainable'. Only relavent if kernel_b is not None.

    """
    def __init__(
        self,
        proj: nn.Module,
        kernel_a: nn.Module = GaussianRBF(trainable=True),
        kernel_b: Optional[nn.Module] = GaussianRBF(trainable=True),
        eps: Union[float, str] = 'trainable'
    ) -> None:
        super().__init__()

        self.kernel_a = kernel_a
        self.kernel_b = kernel_b
        self.proj = proj
        if kernel_b is not None:
            self._init_eps(eps)

    def _init_eps(self, eps: Union[float, str]) -> None:
        if isinstance(eps, float):
            if not 0 < eps < 1:
                raise ValueError("eps should be in (0,1)")
            self.logit_eps = nn.Parameter(torch.tensor(eps).logit(), requires_grad=False)
        elif eps == 'trainable':
            self.logit_eps = nn.Parameter(torch.tensor(0.))
        else:
            raise NotImplementedError("eps should be 'trainable' or a float in (0,1)")

    @property
    def eps(self) -> torch.Tensor:
        return self.logit_eps.sigmoid() if self.kernel_b is not None else torch.tensor(0.)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        similarity = self.kernel_a(self.proj(x), self.proj(y))
        if self.kernel_b is not None:
            similarity = (1-self.eps)*similarity + self.eps*self.kernel_b(x, y)
        return similarity
