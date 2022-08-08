import tensorflow as tf
import numpy as np
from . import distance
from typing import Optional, Union, Callable
from scipy.special import logit


def pseudo_init_fn(x: tf.Tensor, y: tf.Tensor, dist: tf.Tensor) -> tf.Tensor:
    """
    A pseudo-initialization function for the kernel parameter.
    """
    return tf.ones(1, dtype=x.dtype)


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
    The computed bandwidth, `sigma`.
    """
    n = min(x.shape[0], y.shape[0])
    n = n if tf.reduce_all(x[:n] == y[:n]) and x.shape == y.shape else 0
    n_median = n + (tf.math.reduce_prod(dist.shape) - n) // 2 - 1
    sigma = tf.expand_dims((.5 * tf.sort(tf.reshape(dist, (-1,)))[n_median]) ** .5, axis=0)
    return sigma


class BaseKernel(tf.keras.Model):
    """
    The base class for all kernels.
    """
    def __init__(self) -> None:
        super().__init__()
        self.parameter_dict: dict = {}
        self.active_dims: Optional[list] = None
        self.feature_axis: int = -1

    def call(self, x: tf.Tensor, y: tf.Tensor, infer_parameter: bool = False) -> tf.Tensor:
        return NotImplementedError


class SumKernel(tf.keras.Model):
    """
    Construct a kernel by averaging two kernels.

    Parameters:
    ----------
        kernel_a
            the first kernel to be summed.
        kernel_b
            the second kernel to be summed.
    """
    def __init__(
        self,
        kernel_a: BaseKernel,
        kernel_b: BaseKernel
    ) -> None:
        super().__init__()
        self.kernel_a = kernel_a
        self.kernel_b = kernel_b

    def call(self, x: tf.Tensor, y: tf.Tensor, infer_parameter: bool = False) -> tf.Tensor:
        return (self.kernel_a(x, y, infer_parameter) + self.kernel_b(x, y, infer_parameter)) / 2


class ProductKernel(tf.keras.Model):
    """
    Construct a kernel by multiplying two kernels.

    Parameters:
    ----------
        kernel_a
            the first kernel to be summed.
        kernel_b
            the second kernel to be summed.
    """
    def __init__(
        self,
        kernel_a: BaseKernel,
        kernel_b: BaseKernel
    ) -> None:
        super().__init__()
        self.kernel_a = kernel_a
        self.kernel_b = kernel_b

    def call(self, x: tf.Tensor, y: tf.Tensor, infer_parameter: bool = False) -> tf.Tensor:
        return self.kernel_a(x, y, infer_parameter) * self.kernel_b(x, y, infer_parameter)


class GaussianRBF(BaseKernel):
    def __init__(
            self,
            sigma: Optional[tf.Tensor] = None,
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
            The function's signature should match :py:func:`~alibi_detect.utils.tensorflow.kernels.sigma_median`,
            meaning that it should take in the tensors `x`, `y` and `dist` and return `sigma`.
        trainable
            Whether or not to track gradients w.r.t. sigma to allow it to be trained.
        """
        super().__init__()
        self.config = {'sigma': sigma, 'trainable': trainable}
        self.parameter_dict['sigma'] = 'bandwidth'
        if sigma is None:
            self.log_sigma = tf.Variable(np.empty(1), dtype=tf.keras.backend.floatx(), trainable=trainable)
            self.init_required = True
        else:
            sigma = tf.cast(tf.reshape(sigma, (-1,)), dtype=tf.keras.backend.floatx())  # [Ns,]
            self.log_sigma = tf.Variable(tf.math.log(sigma), trainable=trainable)
            self.init_required = False
        self.init_fn_sigma = init_fn_sigma
        self.active_dims = active_dims
        self.feature_axis = feature_axis
        self.trainable = trainable

    @property
    def sigma(self) -> tf.Tensor:
        return tf.math.exp(self.log_sigma)

    def call(self, x: tf.Tensor, y: tf.Tensor, infer_parameter: bool = False) -> tf.Tensor:
        y = tf.cast(y, x.dtype)
        if self.active_dims is not None:
            x = tf.gather(x, self.active_dims, axis=self.feature_axis)
            y = tf.gather(y, self.active_dims, axis=self.feature_axis)
        x, y = tf.reshape(x, (x.shape[0], -1)), tf.reshape(y, (y.shape[0], -1))  # flatten
        dist = distance.squared_pairwise_distance(x, y)  # [Nx, Ny]

        if infer_parameter or self.init_required:
            if self.trainable and infer_parameter:
                raise ValueError("Gradients cannot be computed w.r.t. an inferred sigma value")
            sigma = self.init_fn_sigma(x, y, dist)
            self.log_sigma.assign(tf.math.log(sigma))
            self.init_required = False

        gamma = tf.constant(1. / (2. * self.sigma ** 2), dtype=x.dtype)   # [Ns,]
        # TODO: do matrix multiplication after all?
        kernel_mat = tf.exp(- tf.concat([(g * dist)[None, :, :] for g in gamma], axis=0))  # [Ns, Nx, Ny]
        return tf.reduce_mean(kernel_mat, axis=0)  # [Nx, Ny]

    def get_config(self) -> dict:
        return self.config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class RationalQuadratic(BaseKernel):
    def __init__(
        self,
        alpha: tf.Tensor = None,
        init_fn_alpha: Callable = pseudo_init_fn,
        sigma: tf.Tensor = None,
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
            self.raw_alpha = tf.Variable(np.ones(1), dtype=tf.keras.backend.floatx(), trainable=trainable)
            self.init_required = True
        else:
            self.raw_alpha = tf.cast(tf.reshape(alpha, (-1,)), dtype=tf.keras.backend.floatx())
            self.init_required = False
        if sigma is None:
            self.log_sigma = tf.Variable(np.empty(1), dtype=tf.keras.backend.floatx(), trainable=trainable)
            self.init_required = True
        else:
            sigma = tf.cast(tf.reshape(sigma, (-1,)), dtype=tf.keras.backend.floatx())  # [Ns,]
            self.log_sigma = tf.Variable(tf.math.log(sigma), trainable=trainable)
            self.init_required = False
        self.init_fn_alpha = init_fn_alpha
        self.init_fn_sigma = init_fn_sigma
        self.active_dims = active_dims
        self.feature_axis = feature_axis
        self.trainable = trainable

    @property
    def sigma(self) -> tf.Tensor:
        return tf.math.exp(self.log_sigma)

    @property
    def alpha(self) -> tf.Tensor:
        return self.raw_alpha

    def call(self, x: tf.Tensor, y: tf.Tensor, infer_parameter: bool = False) -> tf.Tensor:
        y = tf.cast(y, x.dtype)
        if self.active_dims is not None:
            x = tf.gather(x, self.active_dims, axis=self.feature_axis)
            y = tf.gather(y, self.active_dims, axis=self.feature_axis)
        x, y = tf.reshape(x, (x.shape[0], -1)), tf.reshape(y, (y.shape[0], -1))
        dist = distance.squared_pairwise_distance(x, y)

        if infer_parameter or self.init_required:
            if self.trainable and infer_parameter:
                raise ValueError("Gradients cannot be computed w.r.t. an inferred sigma value")
            sigma = self.init_fn_sigma(x, y, dist)
            self.log_sigma.assign(tf.math.log(sigma))
            alpha = self.init_fn_alpha(x, y, dist)
            self.raw_alpha.assign(alpha)

        if len(self.sigma) > 1:
            if len(self.sigma) == len(self.alpha):
                kernel_mat = []
                for i in range(len(self.sigma)):
                    kernel_mat.append((1 + tf.square(dist) /
                                       (2 * self.alpha[i] * (self.sigma[i] ** 2))) ** (-self.alpha[i]))
                kernel_mat = tf.reduce_mean(tf.stack(kernel_mat, axis=0), axis=0)
            else:
                raise ValueError("Length of sigma and alpha must be equal")
        else:
            kernel_mat = (1 + tf.square(dist) / (2 * self.alpha * (self.sigma ** 2))) ** (-self.alpha)
        return kernel_mat


class Periodic(BaseKernel):
    def __init__(
        self,
        tau: tf.Tensor = None,
        init_fn_tau: Callable = pseudo_init_fn,
        sigma: tf.Tensor = None,
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
            self.log_tau = tf.Variable(np.empty(1), trainable=trainable)
            self.init_required = True
        else:
            tau = tf.cast(tf.reshape(tau, (-1,)), dtype=tf.keras.backend.floatx())
            self.log_tau = tf.Variable(tf.math.log(tau), trainable=trainable)
            self.init_required = False
        if sigma is None:
            self.log_sigma = tf.Variable(np.empty(1), dtype=tf.keras.backend.floatx(), trainable=trainable)
            self.init_required = True
        else:
            sigma = tf.cast(tf.reshape(sigma, (-1,)), dtype=tf.keras.backend.floatx())  # [Ns,]
            self.log_sigma = tf.Variable(tf.math.log(sigma), trainable=trainable)
            self.init_required = False
        self.init_fn_tau = init_fn_tau
        self.init_fn_sigma = init_fn_sigma
        self.active_dims = active_dims
        self.feature_axis = feature_axis
        self.trainable = trainable

    @property
    def tau(self) -> tf.Tensor:
        return tf.math.exp(self.log_tau)

    @property
    def sigma(self) -> tf.Tensor:
        return tf.math.exp(self.log_sigma)

    def call(self, x: tf.Tensor, y: tf.Tensor, infer_parameter: bool = False) -> tf.Tensor:
        y = tf.cast(y, x.dtype)
        if self.active_dims is not None:
            x = tf.gather(x, self.active_dims, axis=self.feature_axis)
            y = tf.gather(y, self.active_dims, axis=self.feature_axis)
        x, y = tf.reshape(x, (x.shape[0], -1)), tf.reshape(y, (y.shape[0], -1))
        dist = distance.squared_pairwise_distance(x, y)

        if infer_parameter or self.init_required:
            if self.trainable and infer_parameter:
                raise ValueError("Gradients cannot be computed w.r.t. an inferred sigma value")
            sigma = self.init_fn_sigma(x, y, dist)
            self.log_sigma.assign(tf.math.log(sigma))
            tau = self.init_fn_tau(x, y, dist)
            self.log_tau.assign(tf.math.log(tau))

        if len(self.sigma) > 1:
            if len(self.sigma) == len(self.tau):
                kernel_mat = []
                for i in range(len(self.sigma)):
                    kernel_mat.append(tf.math.exp(-2 * tf.square(
                        tf.math.sin(tf.cast(np.pi, x.dtype) * dist / self.tau[i])) / (self.sigma[i] ** 2)))
                kernel_mat = tf.reduce_mean(tf.stack(kernel_mat, axis=0), axis=0)
            else:
                raise ValueError("Length of sigma and alpha must be equal")
        else:
            kernel_mat = tf.math.exp(-2 * tf.square(
                tf.math.sin(tf.cast(np.pi, x.dtype) * dist / self.tau)) / (self.sigma ** 2))
        return kernel_mat


class LocalPeriodic(BaseKernel):
    def __init__(
        self,
        tau: tf.Tensor = None,
        init_fn_tau: Callable = pseudo_init_fn,
        sigma: tf.Tensor = None,
        init_fn_sigma: Callable = sigma_median,
        trainable: bool = False,
        active_dims: Optional[list] = None
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
            self.log_tau = tf.Variable(np.empty(1), trainable=trainable)
            self.init_required = True
        else:
            tau = tf.cast(tf.reshape(tau, (-1,)), dtype=tf.keras.backend.floatx())
            self.log_tau = tf.Variable(tf.math.log(tau), trainable=trainable)
            self.init_required = False
        if sigma is None:
            self.log_sigma = tf.Variable(np.empty(1), dtype=tf.keras.backend.floatx(), trainable=trainable)
            self.init_required = True
        else:
            sigma = tf.cast(tf.reshape(sigma, (-1,)), dtype=tf.keras.backend.floatx())  # [Ns,]
            self.log_sigma = tf.Variable(tf.math.log(sigma), trainable=trainable)
            self.init_required = False
        self.init_fn_tau = init_fn_tau
        self.init_fn_sigma = init_fn_sigma
        self.active_dims = active_dims
        self.trainable = trainable

    @property
    def tau(self) -> tf.Tensor:
        return tf.math.exp(self.log_tau)

    @property
    def sigma(self) -> tf.Tensor:
        return tf.math.exp(self.log_sigma)

    def call(self, x: tf.Tensor, y: tf.Tensor, infer_parameter: bool = False) -> tf.Tensor:
        y = tf.cast(y, x.dtype)
        if self.active_dims is not None:
            x = tf.gather(x, self.active_dims, axis=self.feature_axis)
            y = tf.gather(y, self.active_dims, axis=self.feature_axis)
        x, y = tf.reshape(x, (x.shape[0], -1)), tf.reshape(y, (y.shape[0], -1))
        dist = distance.squared_pairwise_distance(x, y)

        if infer_parameter or self.init_required:
            if self.trainable and infer_parameter:
                raise ValueError("Gradients cannot be computed w.r.t. an inferred sigma value")
            sigma = self.init_fn_sigma(x, y, dist)
            self.log_sigma.assign(tf.math.log(sigma))
            tau = self.init_fn_tau(x, y, dist)
            self.log_tau.assign(tf.math.log(tau))

        if len(self.sigma) > 1:
            if len(self.sigma) == len(self.tau):
                kernel_mat = []
                for i in range(len(self.sigma)):
                    kernel_mat.append(tf.math.exp(-2 * tf.square(
                        tf.math.sin(tf.cast(np.pi, x.dtype) * dist / self.tau[i])) / (self.sigma[i] ** 2)) *
                                      tf.math.exp(-0.5 * tf.square(dist / self.tau[i])))
                kernel_mat = tf.reduce_mean(tf.stack(kernel_mat, axis=0), axis=0)
            else:
                raise ValueError("Length of sigma and alpha must be equal")
        else:
            kernel_mat = tf.math.exp(-2 * tf.square(
                tf.math.sin(tf.cast(np.pi, x.dtype) * dist / self.tau)) / (self.sigma ** 2)) * \
                    tf.math.exp(-0.5 * tf.square(dist / self.tau))
        return kernel_mat


class DeepKernel(tf.keras.Model):
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
        kernel_a: Union[tf.keras.Model, str] = 'rbf',
        kernel_b: Optional[Union[tf.keras.Model, str]] = 'rbf',
        eps: Union[float, str] = 'trainable'
    ) -> None:
        super().__init__()
        if kernel_a == 'rbf':
            kernel_a = GaussianRBF(trainable=True)
        if kernel_b == 'rbf':
            kernel_b = GaussianRBF(trainable=True)
        self.config = {'proj': proj, 'kernel_a': kernel_a, 'kernel_b': kernel_b, 'eps': eps}
        self.kernel_a = kernel_a
        self.kernel_b = kernel_b
        self.proj = proj
        if kernel_b is not None:
            self._init_eps(eps)

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

    def call(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        similarity = self.kernel_a(self.proj(x), self.proj(y))  # type: ignore
        if self.kernel_b is not None:
            similarity = (1-self.eps)*similarity + self.eps*self.kernel_b(x, y)  # type: ignore
        return similarity

    def get_config(self) -> dict:
        return self.config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
