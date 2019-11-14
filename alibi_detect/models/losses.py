import tensorflow as tf
from tensorflow.keras.layers import Flatten
from tensorflow.keras.losses import kld
import tensorflow_probability as tfp
from alibi_detect.models.gmm import gmm_params, gmm_energy


def elbo(y_true: tf.Tensor,
         y_pred: tf.Tensor,
         cov_full: tf.Tensor = None,
         cov_diag: tf.Tensor = None,
         sim: float = .05
         ) -> tf.Tensor:
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


def loss_aegmm(x_true: tf.Tensor,
               x_pred: tf.Tensor,
               z: tf.Tensor,
               gamma: tf.Tensor,
               w_energy: float = .1,
               w_cov_diag: float = .005
               ) -> tf.Tensor:
    """
    Loss function used for OutlierAEGMM.

    Parameters
    ----------
    x_true
        Batch of instances.
    x_pred
        Batch of reconstructed instances by the autoencoder.
    z
        Latent space values.
    gamma
        Membership prediction for mixture model components.
    w_energy
        Weight on sample energy loss term.
    w_cov_diag
        Weight on covariance regularizing loss term.

    Returns
    -------
    Loss value.
    """
    recon_loss = tf.reduce_mean((x_true - x_pred) ** 2)
    phi, mu, cov, L, log_det_cov = gmm_params(z, gamma)
    sample_energy, cov_diag = gmm_energy(z, phi, mu, cov, L, log_det_cov, return_mean=True)
    loss = recon_loss + w_energy * sample_energy + w_cov_diag * cov_diag
    return loss


def loss_vaegmm(x_true: tf.Tensor,
                x_pred: tf.Tensor,
                z: tf.Tensor,
                gamma: tf.Tensor,
                w_recon: float = 1e-7,
                w_energy: float = .1,
                w_cov_diag: float = .005,
                cov_full: tf.Tensor = None,
                cov_diag: tf.Tensor = None,
                sim: float = .05
                ) -> tf.Tensor:
    """
    Loss function used for OutlierVAEGMM.

    Parameters
    ----------
    x_true
        Batch of instances.
    x_pred
        Batch of reconstructed instances by the variational autoencoder.
    z
        Latent space values.
    gamma
        Membership prediction for mixture model components.
    w_recon
        Weight on elbo loss term.
    w_energy
        Weight on sample energy loss term.
    w_cov_diag
        Weight on covariance regularizing loss term.
    cov_full
        Full covariance matrix.
    cov_diag
        Diagonal (variance) of covariance matrix.
    sim
        Scale identity multiplier.

    Returns
    -------
    Loss value.
    """
    recon_loss = elbo(x_true, x_pred, cov_full=cov_full, cov_diag=cov_diag, sim=sim)
    phi, mu, cov, L, log_det_cov = gmm_params(z, gamma)
    sample_energy, cov_diag = gmm_energy(z, phi, mu, cov, L, log_det_cov)
    loss = w_recon * recon_loss + w_energy * sample_energy + w_cov_diag * cov_diag
    return loss


def loss_adv_vae(x_true: tf.Tensor,
                 x_pred: tf.Tensor,
                 model: tf.keras.Model = None,
                 w_model: float = 1.,
                 w_recon: float = 0.,
                 cov_full: tf.Tensor = None,
                 cov_diag: tf.Tensor = None,
                 sim: float = .05
                 ) -> tf.Tensor:
    """
    Loss function used for AdversarialVAE.

    Parameters
    ----------
    x_true
        Batch of instances.
    x_pred
        Batch of reconstructed instances by the variational autoencoder.
    model
        A trained tf.keras model with frozen layers (layers.trainable = False).
    w_model
        Weight on model prediction loss term.
    w_recon
        Weight on elbo loss term.
    cov_full
        Full covariance matrix.
    cov_diag
        Diagonal (variance) of covariance matrix.
    sim
        Scale identity multiplier.

    Returns
    -------
    Loss value.
    """
    y_true = model(x_true)
    y_pred = model(x_pred)
    loss = w_model * tf.reduce_mean(kld(y_true, y_pred))
    if w_recon > 0.:
        loss += w_recon * elbo(x_true, x_pred, cov_full=cov_full, cov_diag=cov_diag, sim=sim)
    return loss
