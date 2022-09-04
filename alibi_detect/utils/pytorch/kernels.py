import numpy as np
import torch
from torch import nn
from . import distance
from typing import Optional, Union, Callable
from copy import deepcopy


def infer_kernel_parameter(kernel, x, y, dist, infer_parameter):
    """
    Infer the kernel parameter from the data.

    Parameters
    ----------
    kernel
        The kernel function.
    x
        Tensor of instances with dimension [Nx, features].
    y
        Tensor of instances with dimension [Ny, features].
    dist
        Tensor with dimensions [Nx, Ny], containing the pairwise distances between `x` and `y`.
    infer_parameter
        Whether to infer the kernel parameter.
    """
    if kernel.trainable and infer_parameter:
        raise ValueError("Gradients cannot be computed w.r.t. an inferred sigma value")
    for parameter in kernel.parameter_dict.values():
        if parameter.requires_init:
            if parameter.init_fn is not None:
                with torch.no_grad():
                    parameter.value.data = parameter.init_fn(x, y, dist).reshape(-1)
            parameter.requires_init = False
    kernel.init_required = False


def sigma_median(x: torch.Tensor, y: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    """
    Bandwidth estimation using the median heuristic :cite:t:`Gretton2012`.

    Parameters
    ----------
    x
        Tensor of instances with dimension [Nx, features].
    y
        Tensor of instances with dimension [Ny, features].
    dist
        Tensor with dimensions [Nx, Ny], containing the pairwise distances between `x` and `y`.

    Returns
    -------
    The logrithm of the computed bandwidth, `log-sigma`.
    """
    n = min(x.shape[0], y.shape[0])
    n = n if (x[:n] == y[:n]).all() and x.shape == y.shape else 0
    n_median = n + (np.prod(dist.shape) - n) // 2 - 1
    sigma = (.5 * dist.flatten().sort().values[n_median].unsqueeze(dim=-1)) ** .5
    return sigma.log()


class KernelParameter:
    """
    Parameter class for kernels.
    """

    def __init__(
        self,
        value: torch.Tensor = None,
        init_fn: Optional[Callable] = None,
        requires_grad: bool = False,
        requires_init: bool = False
    ) -> None:
        super().__init__()
        self.value = nn.Parameter(value if value is not None else torch.ones(1), 
                                  requires_grad=requires_grad)
        self.init_fn = init_fn
        self.requires_init = requires_init


class BaseKernel(nn.Module):
    """
    The base class for all kernels.
    """
    def __init__(self, active_dims: list = None, feature_axis: int = -1) -> None:
        super().__init__()
        self.parameter_dict: dict = {}
        if active_dims is not None:
            self.active_dims = torch.as_tensor(active_dims)
        else:
            self.active_dims = None
        self.feature_axis = feature_axis
        self.init_required = False

    def kernel_function(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor],
                        infer_parameter: bool = False) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor],
                infer_parameter: bool = False) -> torch.Tensor:
        x, y = torch.as_tensor(x), torch.as_tensor(y)
        if self.active_dims is not None:
            x = torch.index_select(x, self.feature_axis, self.active_dims)
            y = torch.index_select(y, self.feature_axis, self.active_dims)
        return self.kernel_function(x, y, infer_parameter)
    
    def __add__(self, other: nn.Module) -> nn.Module:
        if hasattr(other, 'kernel_list'):
            other.kernel_list.append(self)
            return other
        else:
            sum_kernel = SumKernel()
            sum_kernel.kernel_list.append(self)
            sum_kernel.kernel_list.append(other)
            return sum_kernel

    def __radd__(self, other:nn.Module) -> nn.Module:
        return self.__add__(other)

    def __mul__(self, other: nn.Module) -> nn.Module:
        if hasattr(other, 'kernel_factors'):
            other.kernel_factors.append(self)
            return other
        elif hasattr(other, 'kernel_list'):
            sum_kernel = SumKernel()
            for k in other.kernel_list:
                sum_kernel.kernel_list.append(self * k)
            return sum_kernel
        else:
            prod_kernel = ProductKernel()
            prod_kernel.kernel_factors.append(self)
            prod_kernel.kernel_factors.append(other)
            return prod_kernel

    def __rmul__(self, other:nn.Module) -> nn.Module:
        return self.__mul__(other)


class SumKernel(nn.Module):
    """
    Construct a kernel by summing different kernels.

    Parameters:
    ----------------
    """
    def __init__(self) -> None:
        super().__init__()
        self.kernel_list = []

    def forward(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor],
                infer_parameter: bool = False) -> torch.Tensor:
        value_list = []
        for k in self.kernel_list:
            if callable(k):
                value_list.append(k(x, y, infer_parameter))
            else:
                value_list.append(k * torch.ones((x.shape[0], y.shape[0])))
        return torch.sum(torch.stack(value_list), dim=0)
    
    def __add__(self, other: nn.Module) -> nn.Module:
        if hasattr(other, 'kernel_list'):
            for k in other.kernel_list:
                self.kernel_list.append(k)
        else:
            self.kernel_list.append(other)
            return self
    
    def __radd__(self, other:nn.Module) -> nn.Module:
        return self.__add__(other)

    def __mul__(self, other: nn.Module) -> nn.Module:
        if hasattr(other, 'kernel_list'):
            sum_kernel = SumKernel()
            for ki in self.kernel_list:
                for kj in other.kernel_list:
                    sum_kernel.kernel_list.append(ki * kj)
            return sum_kernel
        elif hasattr(other, 'kernel_factors'):
            return other * self
        else:
            sum_kernel = SumKernel()
            for ki in self.kernel_list:
                sum_kernel.kernel_list.append(other * ki)
            return sum_kernel

    def __rmul__(self, other:nn.Module) -> nn.Module:
        return self.__mul__(other)


class ProductKernel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kernel_factors = []

    def forward(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor],
                infer_parameter: bool = False) -> torch.Tensor:
        value_list = []
        for k in self.kernel_factors:
            if callable(k):
                value_list.append(k(x, y, infer_parameter))
            else:
                value_list.append(k * torch.ones((x.shape[0], y.shape[0])))
        return torch.prod(torch.stack(value_list), dim=0)
    
    def __add__(self, other: nn.Module) -> nn.Module:
        if hasattr(other, 'kernel_list'):
            other.kernel_list.append(self)
            return other
        else:
            sum_kernel = SumKernel()
            sum_kernel.kernel_list.append(self)
            sum_kernel.kernel_list.append(other)
            return sum_kernel

    def __radd__(self, other:nn.Module) -> nn.Module:
        return self.__add__(other)

    def __mul__(self, other: nn.Module) -> nn.Module:
        if hasattr(other, 'kernel_list'):
            sum_kernel = SumKernel()
            for k in other.kernel_list:
                tmp_prod_kernel = deepcopy(self)
                tmp_prod_kernel.kernel_factors.append(k)
                sum_kernel.kernel_list.append(tmp_prod_kernel)
            return sum_kernel
        elif hasattr(other, 'kernel_factors'):
            for k in other.kernel_factors:
                self.kernel_factors.append(k)
                return self
        else:
            self.kernel_factors.append(other)
            return self

    def __rmul__(self, other:nn.Module) -> nn.Module:
        return self.__mul__(other)


class GaussianRBF(BaseKernel):
    def __init__(
       self,
       sigma: Optional[torch.Tensor] = None,
       init_fn_sigma: Callable = sigma_median,
       trainable: bool = False,
       active_dims: list = None,
       feature_axis: int = -1
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
        super().__init__(active_dims, feature_axis)
        self.parameter_dict['log-sigma'] = KernelParameter(
                value=sigma.log().reshape(-1) if sigma is not None else None,
                init_fn=init_fn_sigma,
                requires_grad=trainable,
                requires_init=True if sigma is None else False,
                )
        self.trainable = trainable
        self.init_required = any([param.requires_init for param in self.parameter_dict.values()])

    @property
    def sigma(self) -> torch.Tensor:
        return self.parameter_dict['log-sigma'].value.exp()

    def kernel_function(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor],
                        infer_parameter: bool = False) -> torch.Tensor:

        x, y = torch.as_tensor(x), torch.as_tensor(y)
        dist = distance.squared_pairwise_distance(x.flatten(1), y.flatten(1))  # [Nx, Ny]

        if infer_parameter or self.init_required:
            infer_kernel_parameter(self, x, y, dist, infer_parameter)

        gamma = 1. / (2. * self.sigma ** 2)   # [Ns,]
        # TODO: do matrix multiplication after all?
        kernel_mat = torch.exp(- torch.cat([(g * dist)[None, :, :] for g in gamma], dim=0))  # [Ns, Nx, Ny]
        return kernel_mat.mean(dim=0)  # [Nx, Ny]


class RationalQuadratic(BaseKernel):
    def __init__(
        self,
        alpha: torch.Tensor = None,
        init_fn_alpha: Callable = None,
        sigma: torch.Tensor = None,
        init_fn_sigma: Callable = sigma_median,
        trainable: bool = False,
        active_dims: list = None, 
        feature_axis: int = -1 
    ) -> None:
        """
        Rational Quadratic kernel: k(x,y) = (1 + ||x-y||^2 / (2*sigma^2))^(-alpha).
        A forward pass takesa batch of instances x [Nx, features] and y [Ny, features]
        and returns the kernel matrix [Nx, Ny].

        Parameters
        ----------
        alpha
            Exponent parameter of the kernel.
        sigma
            Bandwidth used for the kernel.
        """
        super().__init__(active_dims, feature_axis)
        self.parameter_dict['alpha'] = KernelParameter(
            value=alpha.reshape(-1) if alpha is not None else None,
            init_fn=init_fn_alpha,
            requires_grad=trainable,
            requires_init=True if alpha is None else False
        )
        self.parameter_dict['log-sigma'] = KernelParameter(
            value=sigma.log().reshape(-1) if sigma is not None else None,
            init_fn=init_fn_sigma,
            requires_grad=trainable,
            requires_init=True if sigma is None else False
        )
        self.trainable = trainable
        self.init_required = any([param.requires_init for param in self.parameter_dict.values()])

    @property
    def alpha(self) -> torch.Tensor:
        return self.parameter_dict['alpha'].value

    @property
    def sigma(self) -> torch.Tensor:
        return self.parameter_dict['log-sigma'].value.exp()

    def kernel_function(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor],
                        infer_parameter: bool = False) -> torch.Tensor:

        x, y = torch.as_tensor(x), torch.as_tensor(y)
        dist = distance.squared_pairwise_distance(x.flatten(1), y.flatten(1))

        if infer_parameter or self.init_required:
            infer_kernel_parameter(self, x, y, dist, infer_parameter)

        kernel_mat = torch.stack([(1 + torch.square(dist) /
                                   (2 * self.alpha[i] * (self.sigma[i] ** 2)))
                                  ** (-self.alpha[i]) for i in range(len(self.sigma))], dim=0)

        return kernel_mat.mean(dim=0)


class Periodic(BaseKernel):
    def __init__(
        self,
        tau: torch.Tensor = None,
        init_fn_tau: Callable = None,
        sigma: torch.Tensor = None,
        init_fn_sigma: Callable = sigma_median,
        trainable: bool = False,
        active_dims: list = None, 
        feature_axis: int = -1 
    ) -> None:
        """
        Periodic kernel: k(x,y) = exp(-2 * sin(pi * |x - y| / tau)^2 / (sigma^2)).
        A forward pass takesa batch of instances x [Nx, features] and y [Ny, features]
        and returns the kernel matrix [Nx, Ny].

        Parameters
        ----------
        tau
            Period of the periodic kernel.
        sigma
            Bandwidth used for the kernel.
        """
        super().__init__(active_dims, feature_axis)
        self.parameter_dict['log-tau'] = KernelParameter(
            value=tau.log().reshape(-1) if tau is not None else None,
            init_fn=init_fn_tau,
            requires_grad=trainable,
            requires_init=True if tau is None else False
        )
        self.parameter_dict['log-sigma'] = KernelParameter(
            value=sigma.log().reshape(-1) if sigma is not None else None,
            init_fn=init_fn_sigma,
            requires_grad=trainable,
            requires_init=True if sigma is None else False
        )
        self.trainable = trainable
        self.init_required = any([param.requires_init for param in self.parameter_dict.values()])

    @property
    def tau(self) -> torch.Tensor:
        return self.parameter_dict['log-tau'].value.exp()

    @property
    def sigma(self) -> torch.Tensor:
        return self.parameter_dict['log-sigma'].value.exp()

    def kernel_function(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor],
                        infer_parameter: bool = False) -> torch.Tensor:
        x, y = torch.as_tensor(x), torch.as_tensor(y)
        dist = torch.sqrt(distance.squared_pairwise_distance(x.flatten(1), y.flatten(1)))

        if infer_parameter or self.init_required:
            infer_kernel_parameter(self, x, y, dist, infer_parameter)

        kernel_mat = torch.stack([torch.exp(-2 * torch.square(
            torch.sin(torch.as_tensor(np.pi) * dist / self.tau[i])) / (self.sigma[i] ** 2))
                                  for i in range(len(self.sigma))], dim=0)
        return kernel_mat.mean(dim=0)


class ProjKernel(BaseKernel):
    """
    A kernel that combines a raw kernel (e.g. RBF) with a projection function (e.g. deep net) as
    k(x, y) = k(proj(x), proj(y)). A forward pass takes a batch of instances x [Nx, features] and
    y [Ny, features] and returns the kernel matrix [Nx, Ny].
    
    Parameters:
    ----------
    proj
        The projection to be applied to the inputs before applying raw_kernel
    raw_kernel
        The kernel to apply to the projected inputs. Defaults to a Gaussian RBF with trainable bandwidth.
    """
    def __init__(
        self,
        proj: nn.Module,
        raw_kernel: BaseKernel = GaussianRBF(trainable=True),
    ) -> None:
        super().__init__()
        self.proj = proj
        self.raw_kernel = raw_kernel
        self.init_required = False

    def kernel_function(self, x: torch.Tensor, y: torch.Tensor, infer_parameter: bool = False) -> torch.Tensor:
        return self.raw_kernel(self.proj(x), self.proj(y), infer_parameter)


class DeepKernel(BaseKernel):
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
        kernel_a: BaseKernel = GaussianRBF(trainable=True),
        kernel_b: BaseKernel = GaussianRBF(trainable=True),
        eps: Union[float, str] = 'trainable'
    ) -> None:
        super().__init__()
        proj_kernel = ProjKernel(proj=proj, raw_kernel=kernel_a)
        if kernel_b is not None:
            self._init_eps(eps)
            self.comp_kernel = (1-self.logit_eps.sigmoid() )*proj_kernel + self.logit_eps.sigmoid()*kernel_b
        else:
            self.comp_kernel = proj_kernel

    def _init_eps(self, eps: Union[float, str]) -> None:
        if isinstance(eps, float):
            if not 0 < eps < 1:
                raise ValueError("eps should be in (0,1)")
            self.logit_eps = nn.Parameter(torch.tensor(eps).logit(), requires_grad=False)
        elif eps == 'trainable':
            self.logit_eps = nn.Parameter(torch.tensor(0.))
        else:
            raise NotImplementedError("eps should be 'trainable' or a float in (0,1)")

    def kernel_function(self, x: torch.Tensor, y: torch.Tensor, infer_parmeter=False) -> torch.Tensor:
        return self.comp_kernel(x, y, infer_parmeter)
