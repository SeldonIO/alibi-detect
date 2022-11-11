from abc import abstractmethod
import numpy as np
import torch
from torch import nn
from . import distance
from typing import Optional, Union, Callable, List
from copy import deepcopy
from alibi_detect.utils.frameworks import Framework


def infer_kernel_parameter(
    kernel: 'BaseKernel',
    x: torch.Tensor,
    y: torch.Tensor,
    dist: torch.Tensor,
    infer_parameter: bool = True
) -> None:
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
    The computed bandwidth, `log-sigma`.
    """
    n = min(x.shape[0], y.shape[0])
    n = n if (x[:n] == y[:n]).all() and x.shape == y.shape else 0
    n_median = n + (np.prod(dist.shape) - n) // 2 - 1
    sigma = (.5 * dist.flatten().sort().values[int(n_median)].unsqueeze(dim=-1)) ** .5
    return sigma


def log_sigma_median(x: torch.Tensor, y: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
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
    return torch.log(sigma_median(x, y, dist))


class KernelParameter:
    def __init__(
        self,
        value: torch.Tensor = None,
        init_fn: Optional[Callable] = None,
        requires_grad: bool = False,
        requires_init: bool = False
    ) -> None:
        """
        Parameter class for kernels.

        Parameters
        ----------
        value
            The pre-specified value of the parameter.
        init_fn
            The function used to initialize the parameter.
        requires_grad
            Whether the parameter requires gradient.
        requires_init
            Whether the parameter requires initialization.
        """
        super().__init__()
        self.value = nn.Parameter(value if value is not None else torch.ones(1),
                                  requires_grad=requires_grad)
        self.init_fn = init_fn
        self.requires_init = requires_init


class BaseKernel(nn.Module):
    def __init__(self, active_dims: list = None) -> None:
        """
        The base class for all kernels.

        Parameters
        ----------
        active_dims
            Indices of the dimensions of the feature to be used for the kernel. If None, all dimensions are used.
        """
        super().__init__()
        self.parameter_dict: dict = {}
        if active_dims is not None:
            self.active_dims = torch.as_tensor(active_dims)
        else:
            self.active_dims = None
        self.init_required = False

    @abstractmethod
    def kernel_function(self, x: torch.Tensor, y: torch.Tensor,
                        infer_parameter: Optional[bool] = False) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, y: torch.Tensor,
                infer_parameter: bool = False) -> torch.Tensor:
        if self.active_dims is not None:
            x = torch.index_select(x, -1, self.active_dims)
            y = torch.index_select(y, -1, self.active_dims)
        if len(self.parameter_dict) > 0:
            return self.kernel_function(x, y, infer_parameter)
        else:
            return self.kernel_function(x, y)

    def __add__(
        self,
        other: Union['BaseKernel', torch.Tensor]
    ) -> 'SumKernel':
        if isinstance(other, SumKernel):
            other.kernel_list.append(self)
            return other
        elif isinstance(other, (BaseKernel, ProductKernel, torch.Tensor)):
            sum_kernel = SumKernel()
            sum_kernel.kernel_list.append(self)
            sum_kernel.kernel_list.append(other)
            return sum_kernel
        else:
            raise ValueError('Kernels can only added to another kernel or a constant.')

    def __radd__(self, other: 'BaseKernel') -> 'SumKernel':
        return self.__add__(other)

    def __mul__(
        self,
        other: Union['BaseKernel', torch.Tensor]
    ) -> 'BaseKernel':
        if isinstance(other, ProductKernel):
            other.kernel_factors.append(self)
            return other
        elif isinstance(other, SumKernel):
            sum_kernel = SumKernel()
            for k in other.kernel_list:
                sum_kernel.kernel_list.append(self * k)
            return sum_kernel
        else:
            prod_kernel = ProductKernel()
            prod_kernel.kernel_factors.append(self)
            prod_kernel.kernel_factors.append(other)
            return prod_kernel

    def __rmul__(
        self,
        other: 'BaseKernel'
    ) -> 'BaseKernel':
        return self.__mul__(other)

    def __truediv__(self, other: torch.Tensor) -> 'BaseKernel':
        if isinstance(other, torch.Tensor):
            return self.__mul__(1. / other)
        else:
            raise ValueError('Kernels can only be divided by a constant.')

    def __rtruediv__(self, other):
        raise ValueError('Kernels can not be used as divisor.')

    def __sub__(self, other):
        raise ValueError('Kernels do not support subtraction.')

    def __rsub__(self, other):
        raise ValueError('Kernels do not support subtraction.')


class SumKernel(BaseKernel):
    def __init__(self) -> None:
        """
        Construct a kernel by summing different kernels.
        """
        super().__init__()
        self.kernel_list: List[Union[BaseKernel, torch.Tensor]] = []

    def kernel_function(self, x: torch.Tensor, y: torch.Tensor,
                        infer_parameter: bool = False) -> torch.Tensor:
        value_list: List[torch.Tensor] = []
        for k in self.kernel_list:
            if isinstance(k, (BaseKernel, SumKernel, ProductKernel)):
                value_list.append(k(x, y, infer_parameter))
            elif isinstance(k, torch.Tensor):
                value_list.append(k * torch.ones((x.shape[0], y.shape[0])))
            else:
                raise ValueError(type(k) + 'is not supported by SumKernel.')
        return torch.sum(torch.stack(value_list), dim=0)

    def __add__(
        self,
        other: Union[BaseKernel, torch.Tensor]
    ) -> 'SumKernel':
        if isinstance(other, SumKernel):
            for k in other.kernel_list:
                self.kernel_list.append(k)
        else:
            self.kernel_list.append(other)
        return self

    def __radd__(self, other: BaseKernel) -> 'SumKernel':
        return self.__add__(other)

    def __mul__(
        self,
        other: Union[BaseKernel, torch.Tensor]
    ) -> BaseKernel:
        if isinstance(other, SumKernel):
            sum_kernel = SumKernel()
            for ki in self.kernel_list:
                for kj in other.kernel_list:
                    sum_kernel.kernel_list.append((ki * kj))
            return sum_kernel
        elif isinstance(other, ProductKernel):
            return other * self
        elif isinstance(other, BaseKernel) or isinstance(other, torch.Tensor):
            sum_kernel = SumKernel()
            for ki in self.kernel_list:
                sum_kernel.kernel_list.append(other * ki)
            return sum_kernel
        else:
            raise ValueError(type(other) + 'is not supported by SumKernel.')

    def __rmul__(
        self,
        other: BaseKernel
    ) -> BaseKernel:
        return self.__mul__(other)

    def __truediv__(self, other: torch.Tensor) -> BaseKernel:
        if isinstance(other, torch.Tensor):
            return self.__mul__(1 / other)
        else:
            raise ValueError('Kernels can only be divided by a constant.')

    def __rtruediv__(self, other):
        raise ValueError('Kernels can not be used as divisor.')

    def __sub__(self, other):
        raise ValueError('Kernels do not support subtraction.')

    def __rsub__(self, other):
        raise ValueError('Kernels do not support subtraction.')


class ProductKernel(BaseKernel):
    def __init__(self) -> None:
        """
        Construct a kernel by multiplying different kernels.
        """
        super().__init__()
        self.kernel_factors: List[Union[BaseKernel, torch.Tensor]] = []

    def kernel_function(self, x: torch.Tensor, y: torch.Tensor,
                        infer_parameter: bool = False) -> torch.Tensor:
        value_list: List[torch.Tensor] = []
        for k in self.kernel_factors:
            if isinstance(k, BaseKernel) or isinstance(k, SumKernel) or isinstance(k, ProductKernel):
                value_list.append(k(x, y, infer_parameter))
            elif isinstance(k, torch.Tensor):
                value_list.append(k * torch.ones((x.shape[0], y.shape[0])))
            else:
                raise ValueError(type(k) + 'is not supported by ProductKernel.')
        return torch.prod(torch.stack(value_list), dim=0)

    def __add__(
        self,
        other: Union[BaseKernel, torch.Tensor]
    ) -> 'SumKernel':
        if isinstance(other, SumKernel):
            other.kernel_list.append(self)
            return other
        else:
            sum_kernel = SumKernel()
            sum_kernel.kernel_list.append(self)
            sum_kernel.kernel_list.append(other)
            return sum_kernel

    def __radd__(
        self,
        other: BaseKernel
    ) -> 'SumKernel':
        return self.__add__(other)

    def __mul__(
        self,
        other: Union[BaseKernel, torch.Tensor]
    ) -> BaseKernel:
        if isinstance(other, SumKernel):
            sum_kernel = SumKernel()
            for k in other.kernel_list:
                tmp_prod_kernel = deepcopy(self)
                tmp_prod_kernel.kernel_factors.append(k)
                sum_kernel.kernel_list.append(tmp_prod_kernel)
            return sum_kernel
        elif isinstance(other, ProductKernel):
            for k in other.kernel_factors:
                self.kernel_factors.append(k)
            return self
        elif isinstance(other, BaseKernel) or isinstance(other, torch.Tensor):
            self.kernel_factors.append(other)
            return self
        else:
            raise ValueError(type(other) + 'is not supported by ProductKernel.')

    def __rmul__(
        self,
        other: BaseKernel
    ) -> BaseKernel:
        return self.__mul__(other)

    def __truediv__(self, other: torch.Tensor) -> BaseKernel:
        if isinstance(other, torch.Tensor):
            return self.__mul__(1 / other)
        else:
            raise ValueError('Kernels can only be divided by a constant.')

    def __rtruediv__(self, other):
        raise ValueError('Kernels can not be used as divisor.')

    def __sub__(self, other):
        raise ValueError('Kernels do not support subtraction.')

    def __rsub__(self, other):
        raise ValueError('Kernels do not support subtraction.')


class GaussianRBF(BaseKernel):
    def __init__(
       self,
       sigma: Optional[torch.Tensor] = None,
       init_fn_sigma: Optional[Callable] = None,
       trainable: bool = False,
       active_dims: list = None
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
            meaning that it should take in the tensors `x`, `y` and `dist` and return `sigma`. If `None`, it is set to
            :func:`~alibi_detect.utils.pytorch.kernels.sigma_median`.
        trainable
            Whether or not to track gradients w.r.t. `sigma` to allow it to be trained.
        active_dims
            Indices of the dimensions of the feature to be used for the kernel. If None, all dimensions are used.
        feature_axis
            Axis of the feature dimension.
        """
        super().__init__(active_dims)
        init_fn_sigma = log_sigma_median if init_fn_sigma is None else init_fn_sigma
        self.config = {'sigma': sigma, 'trainable': trainable, 'init_sigma_fn': init_fn_sigma}
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

    def kernel_function(self, x: torch.Tensor, y: torch.Tensor,
                        infer_parameter: bool = False) -> torch.Tensor:
        n_x, n_y = x.shape[0], y.shape[0]
        dist = distance.squared_pairwise_distance(x.reshape(n_x, -1), y.reshape(n_y, -1))  # [Nx, Ny]

        if infer_parameter or self.init_required:
            infer_kernel_parameter(self, x, y, dist, infer_parameter)

        gamma = 1. / (2. * self.sigma ** 2)   # [Ns,]
        # TODO: do matrix multiplication after all?
        kernel_mat = torch.exp(- torch.cat([(g * dist)[None, :, :] for g in gamma], dim=0))  # [Ns, Nx, Ny]
        return kernel_mat.mean(dim=0)  # [Nx, Ny]

    def get_config(self) -> dict:
        """
        Returns a serializable config dict (excluding the input_sigma_fn, which is serialized in alibi_detect.saving).
        """
        cfg = self.config.copy()
        if isinstance(cfg['sigma'], torch.Tensor):
            cfg['sigma'] = cfg['sigma'].detach().cpu().numpy().tolist()
        cfg.update({'flavour': Framework.PYTORCH.value})
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


class RationalQuadratic(BaseKernel):
    def __init__(
        self,
        alpha: torch.Tensor = None,
        init_fn_alpha: Callable = None,
        sigma: torch.Tensor = None,
        init_fn_sigma: Callable = log_sigma_median,
        trainable: bool = False,
        active_dims: list = None
    ) -> None:
        """
        Rational Quadratic kernel: k(x,y) = (1 + ||x-y||^2 / (2*sigma^2))^(-alpha).
        A forward pass takesa batch of instances x [Nx, features] and y [Ny, features]
        and returns the kernel matrix [Nx, Ny].

        Parameters
        ----------
        alpha
            Exponent parameter of the kernel.
        init_alpha_fn
            Function used to compute the exponent parameter `alpha`. Used when `alpha` is to be inferred.
        sigma
            Bandwidth used for the kernel.
        init_sigma_fn
            Function used to compute the bandwidth `sigma`. Used when `sigma` is to be inferred.
        trainable
            Whether or not to track gradients w.r.t. `sigma` to allow it to be trained.
        active_dims
            Indices of the dimensions of the feature to be used for the kernel. If None, all dimensions are used.
        """
        super().__init__(active_dims)
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

    def kernel_function(self, x: torch.Tensor, y: torch.Tensor,
                        infer_parameter: bool = False) -> torch.Tensor:
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
        init_fn_sigma: Callable = log_sigma_median,
        trainable: bool = False,
        active_dims: list = None
    ) -> None:
        """
        Periodic kernel: k(x,y) = exp(-2 * sin(pi * |x - y| / tau)^2 / (sigma^2)).
        A forward pass takesa batch of instances x [Nx, features] and y [Ny, features]
        and returns the kernel matrix [Nx, Ny].

        Parameters
        ----------
        tau
            Period of the periodic kernel.
        init_fn_tau
            Function used to compute the period `tau`. Used when `tau` is to be inferred.
        sigma
            Bandwidth used for the kernel.
        init_fn_sigma
            Function used to compute the bandwidth `sigma`. Used when `sigma` is to be inferred.
        trainable
            Whether or not to track gradients w.r.t. `sigma` to allow it to be trained.
        active_dims
            Indices of the dimensions of the feature to be used for the kernel. If None, all dimensions are used.
        feature_axis
            Axis of the feature dimension.
        """
        super().__init__(active_dims)
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

    def kernel_function(self, x: torch.Tensor, y: torch.Tensor,
                        infer_parameter: bool = False) -> torch.Tensor:
        dist = distance.squared_pairwise_distance(x.flatten(1), y.flatten(1))

        if infer_parameter or self.init_required:
            infer_kernel_parameter(self, x, y, dist, infer_parameter)

        kernel_mat = torch.stack([torch.exp(-2 * torch.square(
            torch.sin(torch.as_tensor(np.pi) * dist / self.tau[i])) / (self.sigma[i] ** 2))
                                  for i in range(len(self.sigma))], dim=0)
        return kernel_mat.mean(dim=0)


class ProjKernel(BaseKernel):
    def __init__(
        self,
        proj: nn.Module,
        raw_kernel: BaseKernel = GaussianRBF(trainable=True),
    ) -> None:
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
        super().__init__()
        self.proj = proj
        self.raw_kernel = raw_kernel
        self.init_required = False

    def kernel_function(
        self,
        x: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        infer_parameter: Optional[bool] = False
    ) -> torch.Tensor:
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
        self.config = {'proj': proj, 'kernel_a': kernel_a, 'kernel_b': kernel_b, 'eps': eps}
        proj_kernel = ProjKernel(proj=proj, raw_kernel=kernel_a)
        if kernel_b is not None:
            self._init_eps(eps)
            self.comp_kernel = (1-self.logit_eps.sigmoid())*proj_kernel + self.logit_eps.sigmoid()*kernel_b
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

    def kernel_function(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        infer_parameter: Optional[bool] = False
    ) -> torch.Tensor:
        return self.comp_kernel(x, y, infer_parameter)

    def get_config(self) -> dict:
        return self.config.copy()

    @classmethod
    def from_config(cls, config):
        return cls(**config)
