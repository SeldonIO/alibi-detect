import tensorflow as tf
from tensorflow.keras.layers import Flatten
import tensorflow_probability as tfp
from odcd.models.gmm import gmm_params, gmm_energy


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


def loss_dagmm(x_true: tf.Tensor,
               x_pred: tf.Tensor,
               z: tf.Tensor,
               gamma: tf.Tensor,
               w_energy: float = .1,
               w_cov_diag: float = .005) -> tf.Tensor:
    recon_error = tf.reduce_mean((x_true - x_pred) ** 2)  # TODO: adjust for VAE version
    phi, mu, cov, L, log_det_cov = gmm_params(z, gamma)
    sample_energy, cov_diag = gmm_energy(z, phi, mu, cov, L, log_det_cov)
    loss = recon_error + w_energy * sample_energy + w_cov_diag * cov_diag
    return loss
