import tensorflow as tf
from tensorflow.keras.layers import Flatten
import tensorflow_probability as tfp


def elbo(y_true: tf.Tensor,
         y_pred: tf.Tensor,
         cov_full: tf.Tensor = None,
         cov_diag: tf.Tensor = None,
         sim: float = .05) -> tf.Tensor:
    """
    Compute ELBO loss.

    Parameters
    ----------
    y_true
        Labels.
    y_pred
        Predictions.
    cov_full
        Full covariance matrix.
    cov_diag
        Diagonal (variance) of covariance matrix.
    sim
        Scale identity multiplier.

    Returns
    -------
    ELBO loss value.
    """
    if isinstance(cov_diag, tf.Tensor):
        sim = None

    if isinstance(cov_full, tf.Tensor):
        y_mn = tfp.distributions.MultivariateNormalFullCovariance(Flatten()(y_pred),
                                                                  covariance_matrix=cov_full)
    else:
        y_mn = tfp.distributions.MultivariateNormalDiag(Flatten()(y_pred),
                                                        scale_diag=cov_diag,
                                                        scale_identity_multiplier=sim)
    loss = -tf.reduce_sum(y_mn.log_prob(Flatten()(y_true)))
    return loss
