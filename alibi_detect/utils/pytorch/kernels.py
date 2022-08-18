import numpy as np
import torch
from torch import nn
from . import distance
from typing import Optional, Union, Callable
from alibi_detect.utils.frameworks import Framework


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
    sigma = (.5 * dist.flatten().sort().values[int(n_median)].unsqueeze(dim=-1)) ** .5
    return sigma.log()


class KernelParameter(object):
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
    def __init__(self) -> None:
        super().__init__()
        self.parameter_dict: dict = {}

    def forward(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor],
                infer_parameter: bool = False) -> torch.Tensor:
        raise NotImplementedError


class DimensionSelectKernel(nn.Module):
    """
    Select a subset of the feature diomensions before apply a given kernel.
    """
    def __init__(self, kernel: BaseKernel, active_dims: list, feature_axis: int = -1) -> None:
        super().__init__()
        self.kernel = kernel
        self.active_dims = torch.as_tensor(active_dims)
        self.feature_axis = feature_axis

    def forward(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor],
                infer_parameter: bool = False) -> torch.Tensor:
        x, y = torch.as_tensor(x), torch.as_tensor(y)
        if self.active_dims is not None:
            x = torch.index_select(x, self.feature_axis, self.active_dims)
            y = torch.index_select(y, self.feature_axis, self.active_dims)
        return self.kernel(x, y, infer_parameter)


class AveragedKernel(nn.Module):
    """
    Construct a kernel by averaging two kernels.

    Parameters:
    ----------
        kernel_a
            the first kernel to be averaged.
        kernel_b
            the second kernel to be averaged.
    """
    def __init__(
        self,
        kernel_a: BaseKernel,
        kernel_b: BaseKernel
    ) -> None:
        super().__init__()
        self.kernel_a = kernel_a
        self.kernel_b = kernel_b

    def forward(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor],
                infer_parameter: bool = False) -> torch.Tensor:
        return (self.kernel_a(x, y, infer_parameter) + self.kernel_b(x, y, infer_parameter)) / 2


class ProductKernel(nn.Module):
    """
    Construct a kernel by multiplying two kernels.

    Parameters:
    ----------
        kernel_a
            the first kernel to be multiplied.
        kernel_b
            the second kernel to be multiplied.
    """
    def __init__(
        self,
        kernel_a: BaseKernel,
        kernel_b: BaseKernel
    ) -> None:
        super().__init__()
        self.kernel_a = kernel_a
        self.kernel_b = kernel_b

    def forward(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor],
                infer_parameter: bool = False) -> torch.Tensor:
        return self.kernel_a(x, y, infer_parameter) * self.kernel_b(x, y, infer_parameter)


class GaussianRBF(BaseKernel):
    def __init__(
       self,
       sigma: Optional[torch.Tensor] = None,
       init_fn_sigma: Optional[Callable] = None,
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
            meaning that it should take in the tensors `x`, `y` and `dist` and return `sigma`. If `None`, it is set to
            :func:`~alibi_detect.utils.pytorch.kernels.sigma_median`.
        trainable
            Whether or not to track gradients w.r.t. `sigma` to allow it to be trained.
        """
        super().__init__()
        init_fn_sigma = sigma_median if init_fn_sigma is None else init_fn_sigma
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

    def forward(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor],
                infer_parameter: bool = False) -> torch.Tensor:

        x, y = torch.as_tensor(x), torch.as_tensor(y)
        dist = distance.squared_pairwise_distance(x.flatten(1), y.flatten(1))  # [Nx, Ny]

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
        init_fn_sigma: Callable = sigma_median,
        trainable: bool = False
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
        super().__init__()
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

    def forward(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor],
                infer_parameter: bool = False) -> torch.Tensor:
        x, y = torch.as_tensor(x), torch.as_tensor(y)
        dist = distance.squared_pairwise_distance(x.flatten(1), y.flatten(1))

        if infer_parameter or self.init_required:
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
        super().__init__()
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

    def forward(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor],
                infer_parameter: bool = False) -> torch.Tensor:
        x, y = torch.as_tensor(x), torch.as_tensor(y)
        dist = torch.sqrt(distance.squared_pairwise_distance(x.flatten(1), y.flatten(1)))

        if infer_parameter or self.init_required:
            infer_kernel_parameter(self, x, y, dist, infer_parameter)

        kernel_mat = torch.stack([torch.exp(-2 * torch.square(
            torch.sin(torch.as_tensor(np.pi) * dist / self.tau[i])) / (self.sigma[i] ** 2))
                                  for i in range(len(self.sigma))], dim=0)
        return kernel_mat.mean(dim=0)


class LocalPeriodic(BaseKernel):
    def __init__(
        self,
        tau: torch.Tensor = None,
        init_fn_tau: Callable = None,
        sigma: torch.Tensor = None,
        init_fn_sigma: Callable = sigma_median,
        trainable: bool = False
    ) -> None:
        """
        Local periodic kernel: k(x,y) = k_rbf(x, y) * k_period(x, y).
        A forward pass takesa batch of instances x [Nx, features] and y [Ny, features]
        and returns the kernel matrix [Nx, Ny].

        Parameters
        ----------
        tau
            Period of the periodic kernel.
        sigma
            Bandwidth used for the kernel.
        """
        super().__init__()
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

    def forward(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor],
                infer_parameter: bool = False) -> torch.Tensor:
        x, y = torch.as_tensor(x), torch.as_tensor(y)
        dist = distance.squared_pairwise_distance(x.flatten(1), y.flatten(1))

        if infer_parameter or self.init_required:
            infer_kernel_parameter(self, x, y, dist, infer_parameter)

        kernel_mat = torch.stack([torch.exp(-2 * torch.square(
            torch.sin(torch.as_tensor(np.pi) * dist / self.tau[i])) / (self.sigma[i] ** 2)) *
                                  torch.exp(-0.5 * torch.square(dist / self.tau[i]))
                                  for i in range(len(self.sigma))], dim=0)
        return kernel_mat.mean(dim=0)


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
        kernel_a: Union[nn.Module, str] = 'rbf',
        kernel_b: Optional[Union[nn.Module, str]] = 'rbf',
        eps: Union[float, str] = 'trainable'
    ) -> None:
        super().__init__()
        self.config = {'proj': proj, 'kernel_a': kernel_a, 'kernel_b': kernel_b, 'eps': eps}
        if kernel_a == 'rbf':
            kernel_a = GaussianRBF(trainable=True)
        if kernel_b == 'rbf':
            kernel_b = GaussianRBF(trainable=True)
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
        similarity = self.kernel_a(self.proj(x), self.proj(y))  # type: ignore[operator]
        if self.kernel_b is not None:
            similarity = (1-self.eps)*similarity + self.eps*self.kernel_b(x, y)  # type: ignore[operator]
        return similarity

    def get_config(self) -> dict:
        return self.config.copy()

    @classmethod
    def from_config(cls, config):
        return cls(**config)
