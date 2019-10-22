import tensorflow as tf
from tensorflow.keras.layers import Flatten
import tensorflow_probability as tfp


def elbo(y_true: tf.Tensor,
         y_pred: tf.Tensor,
         sim: float = .05) -> tf.Tensor:
    """
    Compute ELBO loss.

    Parameters
    ----------
    y_true
        Labels.
    y_pred
        Predictions.
    sim
        Scale identity multiplier.

    Returns
    -------
    ELBO loss value.
    """
    y_mnd = tfp.distributions.MultivariateNormalDiag(Flatten()(y_pred),
                                                     scale_identity_multiplier=sim)
    loss = -tf.reduce_sum(y_mnd.log_prob(Flatten()(y_true)))
    return loss
