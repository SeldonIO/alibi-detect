import numpy as np
import torch
from torch import nn
from . import distance
from typing import Optional, Union, Callable


def pseudo_init_fn(x: torch.Tensor, y: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    return torch.ones(1, dtype=x.dtype, device=x.device)


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
    The computed bandwidth, `sigma`.
    """
    n = min(x.shape[0], y.shape[0])
    n = n if (x[:n] == y[:n]).all() and x.shape == y.shape else 0
    n_median = n + (np.prod(dist.shape) - n) // 2 - 1
    sigma = (.5 * dist.flatten().sort().values[n_median].unsqueeze(dim=-1)) ** .5
    return sigma


class BaseKernel(nn.Module):
    """
    The base class for all kernels.
    Args:
        nn (_type_): _description_
    """
    def __init__(self) -> None:
        super().__init__()
        self.parameter_dict: dict = {}
        self.active_dims: Optional[list] = None
        self.feature_axis: int = -1

    def forward(self, x: torch.Tensor, y: torch.Tensor, infer_parameter: bool = False) -> torch.Tensor:
        raise NotImplementedError


class SumKernel(nn.Module):
    """
    Construct a kernel by summing two kernels.
    Args:
        nn (_type_): _description_
    """
    def __init__(
        self,
        kernel_a: BaseKernel,
        kernel_b: BaseKernel
    ) -> None:
        super().__init__()
        self.kernel_a = kernel_a
        self.kernel_b = kernel_b

    def forward(self, x: torch.Tensor, y: torch.Tensor,  infer_parameter: bool = False) -> torch.Tensor:
        return self.kernel_a(x, y, infer_parameter) + self.kernel_b(x, y, infer_parameter)


class ProductKernel(nn.Module):
    """
    Construct a kernel by multiplying two kernels.
    Args:
        nn (_type_): _description_
    """
    def __init__(
        self,
        kernel_a: BaseKernel,
        kernel_b: BaseKernel
    ) -> None:
        super().__init__()
        self.kernel_a = kernel_a
        self.kernel_b = kernel_b

    def forward(self, x: torch.Tensor, y: torch.Tensor,  infer_parameter: bool = False) -> torch.Tensor:
        return self.kernel_a(x, y, infer_parameter) * self.kernel_b(x, y, infer_parameter)


class GaussianRBF(BaseKernel):
    def __init__(
       self,
       sigma: Optional[torch.Tensor] = None,
       init_fn_sigma: Callable = sigma_median,
       trainable: bool = False,
       active_dims: Optional[list] = None,
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
        super().__init__()
        self.parameter_dict['sigma'] = 'bandwidth'
        if sigma is None:
            self.log_sigma = nn.Parameter(torch.empty(1), requires_grad=trainable)
            self.init_required = True
        else:
            sigma = sigma.reshape(-1)
            self.log_sigma = nn.Parameter(sigma.log(), requires_grad=trainable)
            self.init_required = False
        self.init_fn_sigma = init_fn_sigma
        if active_dims is not None:
            self.active_dims = torch.tensor(active_dims)
        else:
            self.active_dims = None
        self.feature_axis = feature_axis
        self.trainable = trainable

    @property
    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp()

    def forward(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor],
                infer_parameter: bool = False) -> torch.Tensor:

        x, y = torch.as_tensor(x), torch.as_tensor(y)
        if self.active_dims is not None:
            x = torch.index_select(x, self.feature_axis, self.active_dims)
            y = torch.index_select(y, self.feature_axis, self.active_dims)
        dist = distance.squared_pairwise_distance(x.flatten(1), y.flatten(1))  # [Nx, Ny]

        if infer_parameter or self.init_required:
            if self.trainable and infer_parameter:
                raise ValueError("Gradients cannot be computed w.r.t. an inferred sigma value")
            sigma = self.init_fn_sigma(x, y, dist)
            with torch.no_grad():
                self.log_sigma.copy_(sigma.log().clone())
            self.init_required = False

        gamma = 1. / (2. * self.sigma ** 2)   # [Ns,]
        # TODO: do matrix multiplication after all?
        kernel_mat = torch.exp(- torch.cat([(g * dist)[None, :, :] for g in gamma], dim=0))  # [Ns, Nx, Ny]
        return kernel_mat.mean(dim=0)  # [Nx, Ny]


class RationalQuadratic(BaseKernel):
    def __init__(
        self,
        alpha: torch.Tensor = None,
        init_fn_alpha: Callable = pseudo_init_fn,
        sigma: torch.Tensor = None,
        init_fn_sigma: Callable = sigma_median,
        trainable: bool = False,
        active_dims: Optional[list] = None,
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
        super().__init__()
        self.parameter_dict['alpha'] = 'exponent'
        self.parameter_dict['sigma'] = 'bandwidth'
        if alpha is None:
            self.raw_alpha = nn.Parameter(torch.empty(1), requires_grad=trainable)
            self.init_required = True
        else:
            self.raw_alpha = nn.Parameter(alpha, requires_grad=trainable)
            self.init_required = False
        if sigma is None:
            self.log_sigma = nn.Parameter(torch.empty(1), requires_grad=trainable)
            self.init_required = True
        else:
            self.log_sigma = nn.Parameter(sigma.log(), requires_grad=trainable)
            self.init_required = False
        self.init_fn_alpha = init_fn_alpha
        self.init_fn_sigma = init_fn_sigma
        if active_dims is not None:
            self.active_dims = torch.tensor(active_dims)
        else:
            self.active_dims = None
        self.feature_axis = feature_axis
        self.trainable = trainable

    @property
    def alpha(self) -> torch.Tensor:
        return self.raw_alpha

    @property
    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp()

    def forward(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], 
                infer_parameter: bool = False) -> torch.Tensor:
        x, y = torch.as_tensor(x), torch.as_tensor(y)
        if self.active_dims is not None:
            x = torch.index_select(x, self.feature_axis, self.active_dims)
            y = torch.index_select(y, self.feature_axis, self.active_dims)
        dist = distance.squared_pairwise_distance(x.flatten(1), y.flatten(1))
        
        if infer_parameter or self.init_required:
            if self.trainable and infer_parameter:
                raise ValueError("Gradients cannot be computed w.r.t. an inferred sigma value")
            sigma = self.init_fn_sigma(x, y, dist)
            alpha = self.init_fn_alpha(x, y, dist)
            with torch.no_grad():
                self.log_sigma.copy_(sigma.log().clone())
                self.raw_alpha.copy_(alpha.clone())
            self.init_required = False
        
        kernel_mat = (1 + torch.square(dist) / (2 * self.alpha * (self.sigma ** 2))) ** (-self.alpha)
        return kernel_mat


class Periodic(BaseKernel):
    def __init__(
        self,
        tau: torch.Tensor = None,
        init_fn_tau: Callable = pseudo_init_fn,
        sigma: torch.Tensor = None,
        init_fn_sigma: Callable = sigma_median,
        trainable: bool = False,
        active_dims: Optional[list] = None,
        feature_axis: int = -1
    ) -> None:
        """
        Periodic kernel: k(x,y) = .
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
        self.parameter_dict['tau'] = 'period'
        self.parameter_dict['sigma'] = 'bandwidth'
        if tau is None:
            self.log_tau = nn.Parameter(torch.empty(1), requires_grad=trainable)
            self.init_required = True
        else:
            self.log_tau = nn.Parameter(tau.log(), requires_grad=trainable)
            self.init_required = False
        if sigma is None:
            self.log_sigma = nn.Parameter(torch.empty(1), requires_grad=trainable)
            self.init_required = True
        else:
            self.log_sigma = nn.Parameter(sigma.log(), requires_grad=trainable)
            self.init_required = False
        self.init_fn_tau = init_fn_tau
        self.init_fn_sigma = init_fn_sigma
        if active_dims is not None:
            self.active_dims = torch.tensor(active_dims)
        else:
            self.active_dims = None
        self.feature_axis = feature_axis
        self.trainable = trainable

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    @property
    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp()

    def forward(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor],
                infer_parameter: bool = False) -> torch.Tensor:
        x, y = torch.as_tensor(x), torch.as_tensor(y)
        if self.active_dims is not None:
            x = torch.index_select(x, self.feature_axis, self.active_dims)
            y = torch.index_select(y, self.feature_axis, self.active_dims)
        dist = torch.sqrt(distance.squared_pairwise_distance(x.flatten(1), y.flatten(1)))
        
        if infer_parameter or self.init_required:
            if self.trainable and infer_parameter:
                raise ValueError("Gradients cannot be computed w.r.t. an inferred sigma value")
            sigma = self.init_fn_sigma(x, y, dist)
            tau = self.init_fn_tau(x, y, dist)
            with torch.no_grad():
                self.log_sigma.copy_(sigma.log().clone())
                self.log_tau.copy_(tau.log().clone())
            self.init_required = False
        
        kernel_mat = torch.exp(-2 * torch.square(
            torch.sin(torch.as_tensor(np.pi) * dist / self.tau)) / (self.sigma ** 2))
        return kernel_mat


class LocalPeriodic(BaseKernel):
    def __init__(
        self,
        tau: torch.Tensor = None,
        init_fn_tau: Callable = pseudo_init_fn,
        sigma: torch.Tensor = None,
        init_fn_sigma: Callable = sigma_median,
        trainable: bool = False,
        active_dims: Optional[list] = None,
        feature_axis: int = -1
    ) -> None:
        """
        Local periodic kernel: k(x,y) = .
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
        self.parameter_dict['tau'] = 'period'
        self.parameter_dict['sigma'] = 'bandwidth'
        if tau is None:
            self.log_tau = nn.Parameter(torch.empty(1), requires_grad=trainable)
            self.init_required = True
        else:
            self.log_tau = nn.Parameter(tau.log(), requires_grad=trainable)
            self.init_required = False
        if sigma is None:
            self.log_sigma = nn.Parameter(torch.empty(1), requires_grad=trainable)
            self.init_required = True
        else:
            self.log_sigma = nn.Parameter(sigma.log(), requires_grad=trainable)
            self.init_required = False
        self.init_fn_tau = init_fn_tau
        self.init_fn_sigma = init_fn_sigma
        if active_dims is not None:
            self.active_dims = torch.tensor(active_dims)
        else:
            self.active_dims = None
        self.feature_axis = feature_axis
        self.trainable = trainable

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    @property
    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp()

    def forward(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor],
                infer_parameter: bool = False) -> torch.Tensor:
        x, y = torch.as_tensor(x), torch.as_tensor(y)
        if self.active_dims is not None:
            x = torch.index_select(x, self.feature_axis, self.active_dims)
            y = torch.index_select(y, self.feature_axis, self.active_dims)
        dist = distance.squared_pairwise_distance(x.flatten(1), y.flatten(1))
        
        if infer_parameter or self.init_required:
            if self.trainable and infer_parameter:
                raise ValueError("Gradients cannot be computed w.r.t. an inferred sigma value")
            sigma = self.init_fn_sigma(x, y, dist)
            tau = self.init_fn_tau(x, y, dist)
            with torch.no_grad():
                self.log_sigma.copy_(sigma.log().clone())
                self.log_tau.copy_(tau.log().clone())
            self.init_required = False
        
        kernel_mat = torch.exp(-2 * torch.square(
            torch.sin(torch.as_tensor(np.pi) * dist / self.tau)) / (self.sigma ** 2)) * \
            torch.exp(-0.5 * torch.square(dist / self.tau))
        return kernel_mat


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
