import tensorflow as tf
from tensorflow.keras.layers import Flatten
import tensorflow_probability as tfp


def elbo(y_true, y_pred, sim=.05):
    y_mnd = tfp.distributions.MultivariateNormalDiag(Flatten()(y_pred),
                                                     scale_identity_multiplier=sim)
    loss = -tf.reduce_sum(y_mnd.log_prob(Flatten()(y_true)))
    return loss
