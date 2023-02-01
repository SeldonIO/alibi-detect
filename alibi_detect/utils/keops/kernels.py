from abc import abstractmethod
from pykeops.torch import LazyTensor
import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Optional, Union, List
from copy import deepcopy


def infer_kernel_parameter(
    kernel: 'BaseKernel',
    x: LazyTensor,
    y: LazyTensor,
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
        LazyTensor of instances with dimension [Nx, 1, features] or [batch_size, Nx, 1, features].
        The singleton dimension is necessary for broadcasting.
    y
        LazyTensor of instances with dimension [1, Ny, features] or [batch_size, 1, Ny, features].
        The singleton dimension is necessary for broadcasting.
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
        K_sum = torch.tensor(0., device=x.device)
        for k in self.kernel_list:
            if isinstance(k, (BaseKernel, SumKernel, ProductKernel)):
                K_sum = K_sum + k(x, y, infer_parameter)
            elif isinstance(k, torch.Tensor):
                K_sum = K_sum + k
            else:
                raise ValueError(type(k) + 'is not supported by SumKernel.')
        return K_sum

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
        K_prod = torch.tensor(1., device=x.device)
        for k in self.kernel_factors:
            if isinstance(k, BaseKernel) or isinstance(k, SumKernel) or isinstance(k, ProductKernel):
                K_prod = K_prod * k(x, y, infer_parameter)
            elif isinstance(k, torch.Tensor):
                K_prod = K_prod * k
            else:
                raise ValueError(type(k) + 'is not supported by ProductKernel.')
        return K_prod

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
       init_sigma_fn: Optional[Callable] = None,
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
        init_sigma_fn = sigma_mean if init_sigma_fn is None else init_sigma_fn
        self.config = {'sigma': sigma, 'trainable': trainable, 'init_sigma_fn': init_sigma_fn}
        self.parameter_dict['log-sigma'] = KernelParameter(
                value=sigma.log().reshape(-1) if sigma is not None else None,
                init_fn=init_sigma_fn,
                requires_grad=trainable,
                requires_init=True if sigma is None else False,
                )
        self.trainable = trainable
        self.init_required = any([param.requires_init for param in self.parameter_dict.values()])

    @property
    def sigma(self) -> torch.Tensor:
        return self.parameter_dict['log-sigma'].value.exp()

    def kernel_function(self, x: torch.Tensor, y: torch.Tensor,
                        infer_parameter: bool = False) -> LazyTensor:
        if len(x.shape) == 3:
            x = LazyTensor(x[:, :, None, :])
        elif len(x.shape) == 2:
            x = LazyTensor(x[:, None, :])
        else:
            raise ValueError('x should be of shape [batch_size, n_instances, features] or [batch_size, features].')

        if len(y.shape) == 3:
            y = LazyTensor(y[:, None, :, :])
        elif len(y.shape) == 2:
            y = LazyTensor(y[None, :, :])
        else:
            raise ValueError('y should be of shape [batch_size, n_instances, features] or [batch_size, features].')

        dist = ((x - y) ** 2).sum(-1)

        if infer_parameter or self.init_required:
            infer_kernel_parameter(self, x, y, dist, infer_parameter)

        gamma = 1. / (2. * self.sigma ** 2)
        gamma = LazyTensor(gamma[None, None, :]) if len(dist.shape) == 2 else LazyTensor(gamma[None, None, None, :])
        kernel_mat = (- gamma * dist).exp()
        if len(dist.shape) < len(gamma.shape):
            kernel_mat = kernel_mat.sum(-1) / len(self.sigma)
        return kernel_mat


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
        kernel_b: Optional[BaseKernel] = GaussianRBF(trainable=True),
        eps: Union[float, str] = 'trainable'
    ) -> None:
        super().__init__()

        self.proj = proj
        self.kernel_a = kernel_a
        self.kernel_b = kernel_b

        if hasattr(self.kernel_a, 'parameter_dict'):
            for param in self.kernel_a.parameter_dict.keys():
                setattr(self, param, self.kernel_a.parameter_dict[param].value)

        self.proj_kernel = ProjKernel(proj=self.proj, raw_kernel=self.kernel_a)
        if kernel_b is not None:
            self._init_eps(eps)
            self.comp_kernel = (1-self.logit_eps.sigmoid())*self.proj_kernel + self.logit_eps.sigmoid()*self.kernel_b
            if hasattr(self.kernel_b, 'parameter_dict'):
                for param in self.kernel_b.parameter_dict.keys():
                    setattr(self, param, self.kernel_b.parameter_dict[param].value)
        else:
            self.comp_kernel = self.proj_kernel

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

    def kernel_function(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        infer_parameter: Optional[bool] = False
    ) -> torch.Tensor:
        return self.comp_kernel(x, y, infer_parameter)
