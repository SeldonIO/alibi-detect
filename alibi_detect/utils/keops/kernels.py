from pykeops.torch import LazyTensor
import torch
import torch.nn as nn
from typing import Callable, Optional, Union
from alibi_detect.utils.frameworks import Framework
from alibi_detect.utils._types import Literal
from copy import deepcopy


def sigma_mean(x: LazyTensor, y: LazyTensor, dist: LazyTensor, n_min: int = 100) -> torch.Tensor:
    """
    Set bandwidth to the mean distance between instances x and y.

    Parameters
    ----------
    x
        LazyTensor of instances with dimension [Nx, 1, features] or [batch_size, Nx, 1, features].
        The singleton dimension is necessary for broadcasting.
    y
        LazyTensor of instances with dimension [1, Ny, features] or [batch_size, 1, Ny, features].
        The singleton dimension is necessary for broadcasting.
    dist
        LazyTensor with dimensions [Nx, Ny] or [batch_size, Nx, Ny] containing the
        pairwise distances between `x` and `y`.
    n_min
        In order to check whether x equals y after squeezing the singleton dimensions, we check if the
        diagonal of the distance matrix (which is a lazy tensor from which the diagonal cannot be directly extracted)
        consists of all zeros. We do this by computing the k-min distances and k-argmin indices over the
        columns of the distance matrix. We then check if the distances on the diagonal of the distance matrix
        are all zero or not. If they are all zero, then we do not use these distances (zeros) when computing
        the mean pairwise distance as bandwidth. If Nx becomes very large, it is advised to set `n_min`
        to a low enough value to avoid OOM issues. By default we set it to 100 instances.

    Returns
    -------
    The computed bandwidth, `sigma`.
    """
    batched = len(dist.shape) == 3
    if not batched:
        nx, ny = dist.shape
        axis = 1
    else:
        batch_size, nx, ny = dist.shape
        axis = 2
    n_mean = nx * ny
    if nx == ny:
        n_min = min(n_min, nx) if isinstance(n_min, int) else nx
        d_min, id_min = dist.Kmin_argKmin(n_min, axis=axis)
        if batched:
            d_min, id_min = d_min[0], id_min[0]  # first instance in permutation test contains the original data
        rows, cols = torch.where(id_min.cpu() == torch.arange(nx)[:, None])
        if (d_min[rows, cols] == 0.).all():
            n_mean = nx * (nx - 1)
    dist_sum = dist.sum(1).sum(1)[0] if batched else dist.sum(1).sum().unsqueeze(-1)
    sigma = (.5 * dist_sum / n_mean) ** .5
    return sigma


class GaussianRBF(nn.Module):
    def __init__(
        self,
        sigma: Optional[torch.Tensor] = None,
        init_sigma_fn: Optional[Callable] = None,
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
        self.config = {'sigma': sigma, 'trainable': trainable, 'init_sigma_fn': init_sigma_fn}
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

    def get_config(self) -> dict:
        """
        Returns a serializable config dict (excluding the input_sigma_fn, which is serialized in alibi_detect.saving).
        """
        cfg = deepcopy(self.config)
        if isinstance(cfg['sigma'], torch.Tensor):
            cfg['sigma'] = cfg['sigma'].detach().cpu().numpy().tolist()
        cfg.update({'flavour': Framework.KEOPS.value})
        return cfg

    @classmethod
    def from_config(cls, config):
        """
        Instantiates a kernel from a config dictionary.

        Parameters
        ----------
        config
            A kernel config dictionary.
        """
        config.pop('flavour')
        return cls(**config)


class DeepKernel(nn.Module):
    def __init__(
        self,
        proj: nn.Module,
        kernel_a: Union[nn.Module, Literal['rbf']] = 'rbf',
        kernel_b: Optional[Union[nn.Module, Literal['rbf']]] = 'rbf',
        eps: Union[float, Literal['trainable']] = 'trainable'
    ) -> None:
        """
        Computes similarities as k(x,y) = (1-eps)*k_a(proj(x), proj(y)) + eps*k_b(x,y).
        A forward pass takes an already projected batch of instances x_proj and y_proj and optionally
        (if k_b is present) a batch of instances x and y and returns the kernel matrix.
        x_proj can be of shape [Nx, 1, features_proj] or [batch_size, Nx, 1, features_proj].
        y_proj can be of shape [1, Ny, features_proj] or [batch_size, 1, Ny, features_proj].
        x can be of shape [Nx, 1, features] or [batch_size, Nx, 1, features].
        y can be of shape [1, Ny, features] or [batch_size, 1, Ny, features].
        The returned kernel matrix can be of shape [Nx, Ny] or [batch_size, Nx, Ny].
        x, y and the returned kernel matrix are all lazy tensors.

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
        super().__init__()
        self.config = {'proj': proj, 'kernel_a': kernel_a, 'kernel_b': kernel_b, 'eps': eps}
        if kernel_a == 'rbf':
            kernel_a = GaussianRBF(trainable=True)
        if kernel_b == 'rbf':
            kernel_b = GaussianRBF(trainable=True)
        self.kernel_a: Callable = kernel_a
        self.kernel_b: Callable = kernel_b
        self.proj = proj
        if kernel_b is not None:
            self._init_eps(eps)

    def _init_eps(self, eps: Union[float, Literal['trainable']]) -> None:
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

    def forward(self, x_proj: LazyTensor, y_proj: LazyTensor, x: Optional[LazyTensor] = None,
                y: Optional[LazyTensor] = None) -> LazyTensor:
        similarity = self.kernel_a(x_proj, y_proj)
        if self.kernel_b is not None:
            similarity = (1-self.eps)*similarity + self.eps*self.kernel_b(x, y)
        return similarity

    def get_config(self) -> dict:
        return deepcopy(self.config)

    @classmethod
    def from_config(cls, config):
        return cls(**config)
