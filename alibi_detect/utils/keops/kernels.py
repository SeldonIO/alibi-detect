from alibi_detect.utils.pytorch.kernels import sigma_median  # TODO: keops sigma_median?
import numpy as np
from pykeops.torch import LazyTensor
import torch
import torch.nn as nn
from typing import Callable, List, Tuple, Union


class GaussianRBF(nn.Module):
    def __init__(
        self,
        sigma: torch.Tensor = None,
        init_sigma_fn: Callable = sigma_median,
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
            The function's signature should match :py:func:`~alibi_detect.utils.pytorch.kernels.sigma_median`,
            meaning that it should take in the tensors `x`, `y` and `dist` and return `sigma`.
        trainable
            Whether or not to track gradients w.r.t. `sigma` to allow it to be trained.
        """
        super().__init__()
        if sigma is None:
            self.log_sigma = nn.Parameter(torch.empty(1), requires_grad=trainable)
            self.init_required = True
        else:
            #sigma = torch.Tensor([sigma])  # TODO: ensure it's done somewhere else
            sigma = sigma.reshape(-1)  # [1]
            self.log_sigma = nn.Parameter(sigma.log(), requires_grad=trainable)
            self.init_required = False
        self.init_sigma_fn = init_sigma_fn
        self.trainable = trainable

        super().__init__()
        if sigma is None:
            self.log_sigma = nn.Parameter(torch.empty(1), requires_grad=trainable)
            self.init_required = True
        else:
            sigma = sigma.reshape(-1)  # [Ns,]  TODO: ensure this works with keops
            self.log_sigma = nn.Parameter(sigma.log(), requires_grad=trainable)
            self.init_required = False
        self.init_sigma_fn = init_sigma_fn
        self.trainable = trainable

    @property
    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp()

    # TODO: could use original kernel with some tweaks?
    #   - LazyTensor input
    #   - permutations input
    #   - reduce_sum done in main detector
    def forward(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor],
                infer_sigma: bool = False, permutations: bool = False, reduce_sum: bool = False) -> LazyTensor:

        x, y = torch.as_tensor(x), torch.as_tensor(y)

        #if isinstance(x, np.ndarray):
        #    x = torch.as_tensor(x)
        #if isinstance(y, np.ndarray):
        #    y = torch.as_tensor(y)

        if not permutations:
            x_i = LazyTensor(x[:, None, :])  # [n, 1, d]
            y_j = LazyTensor(y[None, :, :])  # [1, m, d]
        else:
            x_i = LazyTensor(x[:, :, None, :])  # [perms+1, n, 1, d]
            y_j = LazyTensor(y[:, None, :, :])  # [perms+1, 1, m, d]
        d_ij = ((x_i - y_j) ** 2).sum(-1)  # [n, m]

        if infer_sigma or self.init_required:
            if self.trainable and infer_sigma:
                raise ValueError("Gradients cannot be computed w.r.t. an inferred sigma value")
            sigma = self.init_sigma_fn(x, y, d_ij)  # TODO: would not work with default init fn
            with torch.no_grad():
                self.log_sigma.copy_(sigma.log().clone())
            self.init_required = False

        gamma = 1. / (2. * self.sigma ** 2)   # [1] TODO: [Ns,]?
        if not permutations:
            gamma = LazyTensor(gamma[None, None, :])  # [1, 1, 1]
        else:
            gamma = LazyTensor(gamma[None, None, None, :])  # [1, 1, 1, 1]
        k_ij = (- gamma * d_ij).exp()  # [n, m] or [perms+1, n, m]
        if reduce_sum:
            k_ij = k_ij.sum(1).sum(1).squeeze(-1)   # [1] or [perms+1]
        return k_ij


class DeepKernelKeops(nn.Module):
    def __init__(
            self,
            proj: nn.Module,
            kernel_a: nn.Module = GaussianRBFKeops(trainable=True),
            kernel_b: nn.Module = GaussianRBFKeops(trainable=True),
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

    # TODO: where does this belong?
    def forward_perms(self, x_proj: torch.Tensor, y_proj: torch.Tensor,
                      x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        similarity = self.kernel_a(x_proj, y_proj, permutations=True)  # [perms+1, n, m]
        if self.kernel_b is not None:
            similarity = (1-self.eps)*similarity + self.eps*self.kernel_b(x, y, permutations=True)
        return similarity.sum(1).sum(1).squeeze(-1)   # [perms+1]
