import tensorflow as tf
from . import distance
from typing import Optional


class GaussianRBF:
    def __init__(self, sigma: Optional[tf.Tensor] = None) -> None:
        """
        Gaussian RBF kernel: k(x,y) = exp(-(1/(2*sigma^2)||x-y||^2). A forward pass takes
        a batch of instances x [Nx, features] and y [Ny, features] and returns the kernel
        matrix [Nx, Ny].

        Parameters
        ----------
        sigma
            Optional sigma used for the kernel.
        """
        super().__init__()
        self.sigma = sigma

    def __call__(self, x: tf.Tensor, y: tf.Tensor, infer_sigma: bool = False) -> tf.Tensor:

        dist = distance.squared_pairwise_distance(x, y)  # [Nx, Ny]

        if infer_sigma:
            n = min(x.shape[0], y.shape[0])
            n = n if tf.reduce_all(x[:n] == y[:n]) and x.shape == y.shape else 0
            n_median = n + (tf.math.reduce_prod(dist.shape) - n) // 2 - 1
            self.sigma = tf.expand_dims((.5 * tf.sort(tf.reshape(dist, (-1,)))[n_median]) ** .5, axis=0)

        gamma = 1. / (2. * self.sigma ** 2)   # [Ns,]
        # TODO: do matrix multiplication after all?
        kernel_mat = tf.exp(- tf.concat([(g * dist)[None, :, :] for g in gamma], axis=0))  # [Ns, Nx, Ny]
        return tf.reduce_mean(kernel_mat, axis=0)  # [Nx, Ny]
