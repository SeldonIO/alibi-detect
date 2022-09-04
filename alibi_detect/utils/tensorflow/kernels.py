import tensorflow as tf
import numpy as np
from . import distance
from typing import Optional, Union, Callable
from scipy.special import logit
from copy import deepcopy
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
                parameter.value.assign(tf.reshape(parameter.init_fn(x, y, dist), -1))
            parameter.requires_init = False
    kernel.init_required = False


def sigma_median(x: tf.Tensor, y: tf.Tensor, dist: tf.Tensor) -> tf.Tensor:
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
    n = n if tf.reduce_all(x[:n] == y[:n]) and x.shape == y.shape else 0
    n_median = n + (tf.math.reduce_prod(dist.shape) - n) // 2 - 1
    sigma = tf.expand_dims((.5 * tf.sort(tf.reshape(dist, (-1,)))[n_median]) ** .5, axis=0)
    return tf.math.log(sigma)


class KernelParameter(object):
    """
    Parameter class for kernels.
    """
    def __init__(
        self,
        value: tf.Tensor = None,
        init_fn: Optional[Callable] = None,
        requires_grad: bool = False,
        requires_init: bool = False
    ) -> None:
        self.value = tf.Variable(value if value is not None
                                 else tf.ones(1, dtype=tf.keras.backend.floatx()),
                                 trainable=requires_grad)
        self.init_fn = init_fn
        self.requires_init = requires_init

    def __repr__(self) -> str:
        return self.value.__repr__()


class BaseKernel(tf.keras.Model):
    """
    The base class for all kernels.
    """
    def __init__(self, active_dims: list = None, feature_axis: int = -1) -> None:
        super().__init__()
        self.parameter_dict: dict = {}
        self.active_dims = active_dims
        self.feature_axis = feature_axis
        self.init_required = False

    def kernel_function(self, x: tf.Tensor, y: tf.Tensor, infer_parameter: bool = False) -> tf.Tensor:
        return NotImplementedError

    def call(self, x: tf.Tensor, y: tf.Tensor, infer_parameter: bool = False) -> tf.Tensor:
        y = tf.cast(y, x.dtype)
        if self.active_dims is not None:
            x = tf.gather(x, self.active_dims, axis=self.feature_axis)
            y = tf.gather(y, self.active_dims, axis=self.feature_axis)
        return self.kernel_function(x, y, infer_parameter)

    def __add__(self, other: tf.keras.Model) -> tf.keras.Model:
        if hasattr(other, 'kernel_list'):
            other.kernel_list.append(self)
            return other
        else:
            sum_kernel = SumKernel()
            sum_kernel.kernel_list.append(self)
            sum_kernel.kernel_list.append(other)
            return sum_kernel

    def __radd__(self, other: tf.keras.Model) -> tf.keras.Model:
        return self.__add__(other)

    def __mul__(self, other: tf.keras.Model) -> tf.keras.Model:
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

    def __rmul__(self, other: tf.keras.Model) -> tf.keras.Model:
        return self.__mul__(other)


class SumKernel(tf.keras.Model):
    """
    Construct a kernel by summing different kernels.

    Parameters:
    ----------------
    """
    def __init__(self) -> None:
        super().__init__()
        self.kernel_list = []

    def call(self, x: Union[np.ndarray, tf.Tensor], y: Union[np.ndarray, tf.Tensor],
             infer_parameter: bool = False) -> tf.Tensor:
        value_list = []
        for k in self.kernel_list:
            if callable(k):
                value_list.append(k(x, y, infer_parameter))
            else:
                value_list.append(k * tf.ones((x.shape[0], y.shape[0])))
        return tf.reduce_sum(tf.stack(value_list), axis=0)

    def __add__(self, other: tf.keras.Model) -> tf.keras.Model:
        if hasattr(other, 'kernel_list'):
            for k in other.kernel_list:
                self.kernel_list.append(k)
        else:
            self.kernel_list.append(other)
            return self

    def __radd__(self, other: tf.keras.Model) -> tf.keras.Model:
        return self.__add__(other)

    def __mul__(self, other: tf.keras.Model) -> tf.keras.Model:
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

    def __rmul__(self, other: tf.keras.Model) -> tf.keras.Model:
        return self.__mul__(other)


class ProductKernel(tf.keras.Model):
    def __init__(self) -> None:
        super().__init__()
        self.kernel_factors = []

    def call(self, x: Union[np.ndarray, tf.Tensor], y: Union[np.ndarray, tf.Tensor],
             infer_parameter: bool = False) -> tf.Tensor:
        value_list = []
        for k in self.kernel_factors:
            if callable(k):
                value_list.append(k(x, y, infer_parameter))
            else:
                value_list.append(k * tf.ones((x.shape[0], y.shape[0])))
        return tf.reduce_prod(tf.stack(value_list), axis=0)

    def __add__(self, other: tf.keras.Model) -> tf.keras.Model:
        if hasattr(other, 'kernel_list'):
            other.kernel_list.append(self)
            return other
        else:
            sum_kernel = SumKernel()
            sum_kernel.kernel_list.append(self)
            sum_kernel.kernel_list.append(other)
            return sum_kernel

    def __radd__(self, other: tf.keras.Model) -> tf.keras.Model:
        return self.__add__(other)

    def __mul__(self, other: tf.keras.Model) -> tf.keras.Model:
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

    def __rmul__(self, other: tf.keras.Model) -> tf.keras.Model:
        return self.__mul__(other)


class GaussianRBF(BaseKernel):
    def __init__(
            self,
            sigma: Optional[tf.Tensor] = None,
            init_fn_sigma: Optional[Callable] = None,
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
            The function's signature should match :py:func:`~alibi_detect.utils.tensorflow.kernels.sigma_median`,
            meaning that it should take in the tensors `x`, `y` and `dist` and return `sigma`. If `None`, it is set to
            :func:`~alibi_detect.utils.tensorflow.kernels.sigma_median`.
        trainable
            Whether or not to track gradients w.r.t. sigma to allow it to be trained.
        """
        super().__init__(active_dims, feature_axis)
        init_fn_sigma = sigma_median if init_fn_sigma is None else init_fn_sigma
        self.config = {'sigma': sigma, 'trainable': trainable, 'init_sigma_fn': init_fn_sigma}
        self.parameter_dict['log-sigma'] = KernelParameter(
            value=tf.reshape(tf.math.log(
                tf.cast(sigma, tf.keras.backend.floatx())), -1) if sigma is not None else None,
            init_fn=init_fn_sigma,
            requires_grad=trainable,
            requires_init=True if sigma is None else False
        )
        self.trainable = trainable
        self.init_required = any([param.requires_init for param in self.parameter_dict.values()])

    @property
    def sigma(self) -> tf.Tensor:
        return tf.math.exp(self.parameter_dict['log-sigma'].value)

    def kernel_function(self, x: tf.Tensor, y: tf.Tensor, infer_parameter: bool = False) -> tf.Tensor:
        y = tf.cast(y, x.dtype)
        x, y = tf.reshape(x, (x.shape[0], -1)), tf.reshape(y, (y.shape[0], -1))  # flatten
        dist = distance.squared_pairwise_distance(x, y)  # [Nx, Ny]

        if infer_parameter or self.init_required:
            infer_kernel_parameter(self, x, y, dist, infer_parameter)

        gamma = tf.constant(1. / (2. * self.sigma ** 2), dtype=x.dtype)   # [Ns,]
        # TODO: do matrix multiplication after all?
        kernel_mat = tf.exp(- tf.concat([(g * dist)[None, :, :] for g in gamma], axis=0))  # [Ns, Nx, Ny]
        return tf.reduce_mean(kernel_mat, axis=0)  # [Nx, Ny]

    def get_config(self) -> dict:
        """
        Returns a serializable config dict (excluding the input_sigma_fn, which is serialized in alibi_detect.saving).
        """
        cfg = self.config.copy()
        if isinstance(cfg['sigma'], tf.Tensor):
            cfg['sigma'] = cfg['sigma'].numpy().tolist()
        cfg.update({'flavour': Framework.TENSORFLOW.value})
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
        alpha: tf.Tensor = None,
        init_fn_alpha: Callable = None,
        sigma: tf.Tensor = None,
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
            value=tf.reshape(
                tf.cast(alpha, tf.keras.backend.floatx()), -1) if alpha is not None else None,
            init_fn=init_fn_alpha,
            requires_grad=trainable,
            requires_init=True if alpha is None else False
        )
        self.parameter_dict['log-sigma'] = KernelParameter(
            value=tf.reshape(tf.math.log(
                tf.cast(sigma, tf.keras.backend.floatx())), -1) if sigma is not None else None,
            init_fn=init_fn_sigma,
            requires_grad=trainable,
            requires_init=True if sigma is None else False
        )
        self.trainable = trainable
        self.init_required = any([param.requires_init for param in self.parameter_dict.values()])

    @property
    def sigma(self) -> tf.Tensor:
        return tf.math.exp(self.parameter_dict['log-sigma'].value)

    @property
    def alpha(self) -> tf.Tensor:
        return self.parameter_dict['alpha'].value

    def kernel_function(self, x: tf.Tensor, y: tf.Tensor, infer_parameter: bool = False) -> tf.Tensor:
        y = tf.cast(y, x.dtype)
        x, y = tf.reshape(x, (x.shape[0], -1)), tf.reshape(y, (y.shape[0], -1))
        dist = distance.squared_pairwise_distance(x, y)

        if infer_parameter or self.init_required:
            infer_kernel_parameter(self, x, y, dist, infer_parameter)

        kernel_mat = tf.stack([(1 + tf.square(dist) /
                                (2 * self.alpha[i] * (self.sigma[i] ** 2)))
                               ** (-self.alpha[i]) for i in range(len(self.sigma))], axis=0)
        return tf.reduce_mean(kernel_mat, axis=0)


class Periodic(BaseKernel):
    def __init__(
        self,
        tau: tf.Tensor = None,
        init_fn_tau: Callable = None,
        sigma: tf.Tensor = None,
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
            value=tf.reshape(tf.math.log(
                tf.cast(tau, tf.keras.backend.floatx())), -1) if tau is not None else None,
            init_fn=init_fn_tau,
            requires_grad=trainable,
            requires_init=True if tau is None else False
        )
        self.parameter_dict['log-sigma'] = KernelParameter(
            value=tf.reshape(tf.math.log(
                tf.cast(sigma, tf.keras.backend.floatx())), -1) if sigma is not None else None,
            init_fn=init_fn_sigma,
            requires_grad=trainable,
            requires_init=True if sigma is None else False
        )
        self.trainable = trainable
        self.init_required = any([param.requires_init for param in self.parameter_dict.values()])

    @property
    def tau(self) -> tf.Tensor:
        return tf.math.exp(self.parameter_dict['log-tau'].value)

    @property
    def sigma(self) -> tf.Tensor:
        return tf.math.exp(self.parameter_dict['log-sigma'].value)

    def kernel_function(self, x: tf.Tensor, y: tf.Tensor, infer_parameter: bool = False) -> tf.Tensor:
        y = tf.cast(y, x.dtype)
        x, y = tf.reshape(x, (x.shape[0], -1)), tf.reshape(y, (y.shape[0], -1))
        dist = distance.squared_pairwise_distance(x, y)

        if infer_parameter or self.init_required:
            infer_kernel_parameter(self, x, y, dist, infer_parameter)

        kernel_mat = tf.stack([tf.math.exp(-2 * tf.square(
            tf.math.sin(tf.cast(np.pi, x.dtype) * dist / self.tau[i])) / (self.sigma[i] ** 2))
                               for i in range(len(self.sigma))], axis=0)
        return tf.reduce_mean(kernel_mat, axis=0)


class ProjKernel(BaseKernel):
    def __init__(
        self,
        proj: tf.keras.Model,
        raw_kernel: BaseKernel = GaussianRBF(trainable=True),
    ) -> None:
        super().__init__()
        self.proj = proj
        self.raw_kernel = raw_kernel
        self.init_required = False

    def kernel_function(self, x: tf.Tensor, y: tf.Tensor, infer_parameter: bool = False) -> tf.Tensor:
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
        either specified or set to 'trainable'. Only relavent is kernel_b is not None.

    """
    def __init__(
        self,
        proj: tf.keras.Model,
        kernel_a: BaseKernel = GaussianRBF(trainable=True),
        kernel_b: BaseKernel = GaussianRBF(trainable=True),
        eps: Union[float, str] = 'trainable'
    ) -> None:
        super().__init__()
        proj_kernel = ProjKernel(proj=proj, raw_kernel=kernel_a)
        if kernel_b is not None:
            self._init_eps(eps)
            self.comp_kernel = (1-tf.sigmoid(self.logit_eps))*proj_kernel + tf.sigmoid(self.logit_eps)*kernel_b
        else:
            self.comp_kernel = proj_kernel
        self.config = {'proj': proj, 'kernel_a': kernel_a, 'kernel_b': kernel_b, 'eps': eps}

    def _init_eps(self, eps: Union[float, str]) -> None:
        if isinstance(eps, float):
            if not 0 < eps < 1:
                raise ValueError("eps should be in (0,1)")
            eps = tf.constant(eps)
            self.logit_eps = tf.Variable(tf.constant(logit(eps)), trainable=False)
        elif eps == 'trainable':
            self.logit_eps = tf.Variable(tf.constant(0.))
        else:
            raise NotImplementedError("eps should be 'trainable' or a float in (0,1)")

    @property
    def eps(self) -> tf.Tensor:
        return tf.math.sigmoid(self.logit_eps) if self.kernel_b is not None else tf.constant(0.)

    def kernel_function(self, x: tf.Tensor, y: tf.Tensor, infer_parameter: bool = False) -> tf.Tensor:
        return self.comp_kernel(x, y, infer_parameter)

    def get_config(self) -> dict:
        return self.config.copy()

    @classmethod
    def from_config(cls, config):
        return cls(**config)
