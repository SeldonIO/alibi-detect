import tensorflow as tf
import numpy as np
from . import distance
from typing import Optional, Union
from scipy.special import logit


class GaussianRBF(tf.keras.Model):
    def __init__(self, sigma: Optional[tf.Tensor] = None, trainable: bool = False) -> None:
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
        self.config = {'sigma': sigma, 'trainable': trainable}
        if sigma is None:
            self.log_sigma = tf.Variable(np.empty(1), dtype=tf.keras.backend.floatx(), trainable=trainable)
            self.init_required = True
        else:
            sigma = tf.cast(tf.reshape(sigma, (-1,)), dtype=tf.keras.backend.floatx())  # [Ns,]
            self.log_sigma = tf.Variable(tf.math.log(sigma), trainable=trainable)
            self.init_required = False
        self.trainable = trainable

    @property
    def sigma(self) -> tf.Tensor:
        return tf.math.exp(self.log_sigma)

    def call(self, x: tf.Tensor, y: tf.Tensor, infer_sigma: bool = False) -> tf.Tensor:
        y = tf.cast(y, x.dtype)
        x, y = tf.reshape(x, (x.shape[0], -1)), tf.reshape(y, (y.shape[0], -1))  # flatten
        dist = distance.squared_pairwise_distance(x, y)  # [Nx, Ny]

        if infer_sigma or self.init_required:
            if self.trainable and infer_sigma:
                raise ValueError("Gradients cannot be computed w.r.t. an inferred sigma value")
            n = min(x.shape[0], y.shape[0])
            n = n if tf.reduce_all(x[:n] == y[:n]) and x.shape == y.shape else 0
            n_median = n + (tf.math.reduce_prod(dist.shape) - n) // 2 - 1
            sigma = tf.expand_dims((.5 * tf.sort(tf.reshape(dist, (-1,)))[n_median]) ** .5, axis=0)
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
        kernel_a: tf.keras.Model = GaussianRBF(trainable=True),
        kernel_b: Optional[tf.keras.Model] = GaussianRBF(trainable=True),
        eps: Union[float, str] = 'trainable'
    ) -> None:
        super().__init__()
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
        similarity = self.kernel_a(self.proj(x), self.proj(y))
        if self.kernel_b is not None:
            similarity = (1-self.eps)*similarity + self.eps*self.kernel_b(x, y)
        return similarity

    def get_config(self) -> dict:
        return self.config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
