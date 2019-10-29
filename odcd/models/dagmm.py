import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten
from typing import Tuple


def relative_euclidean_distance(x: tf.Tensor, y: tf.Tensor, axis: int = -1):
    # TODO: make sure also works for higher dim enc eg for images
    dist = tf.norm(x - y, ord=2, axis=axis) / tf.norm(x, axis=axis)
    return dist


class DAGMM(tf.keras.Model):
    """  Deep Autoencoding Gaussian Mixture Model.  """

    def __init__(self,
                 encoder_net: tf.keras.Sequential,
                 decoder_net: tf.keras.Sequential,
                 gmm_density_net: tf.keras.Sequential,
                 n_gmm: int,
                 latent_dim: int,  # TODO: infer from encoder_net
                 name: str = 'dagmm') -> None:
        super(DAGMM, self).__init__(name=name)
        self.encoder = encoder_net
        self.decoder = decoder_net
        self.gmm_density = gmm_density_net
        self.n_gmm = n_gmm
        self.latent_dim = latent_dim

    def call(self, x: tf.Tensor) -> tf.Tensor:
        enc = self.encoder(x)
        x_recon = self.decoder(enc)
        # TODO: make sure also works for higher dim enc eg for images; flatten first?!
        # TODO: is rec_cos / rec_euc best representation for error?!
        # TODO: don't flatten for images b/c lose locality info and does not allow to
        # TODO: have conv layers in estimation network
        if len(x.shape) > 2:
            x = Flatten()(x)
            x_recon = Flatten()(x_recon)
        if len(enc.shape) > 2:
            enc = Flatten()(enc)
        rec_cos = tf.reshape(tf.keras.losses.cosine_similarity(x, x_recon, -1), (-1, 1))
        rec_euc = tf.reshape(relative_euclidean_distance(x, x_recon, -1), (-1, 1))
        # TODO: add clipping for rec_euc using tf.clip_by_value b/c could be +inf
        z = tf.concat([enc, rec_cos, rec_euc], -1)
        gamma = self.gmm_density(z)
        # TODO: check whether reshaping before returning is needed for x_recon
        return enc, x_recon, z, gamma

    def gmm_params(self,
                   z: tf.Tensor,
                   gamma: tf.Tensor) \
            -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

        # nb of samples in batch
        N = gamma.shape[0]

        # K
        sum_gamma = tf.reduce_sum(gamma, 0)

        # K
        phi = sum_gamma / N

        # K x D (D = latent_dim)
        mu = (tf.reduce_sum(tf.expand_dims(gamma, -1) * tf.expand_dims(z, 1), 0)
              / tf.expand_dims(sum_gamma, -1))

        # N x K x D
        z_mu = tf.expand_dims(z, 1) - tf.expand_dims(mu, 0)

        # N x K x D x D
        z_mu_outer = tf.expand_dims(z_mu, -1) * tf.expand_dims(z_mu, -2)

        # K x D x D
        cov = (tf.reduce_sum(tf.expand_dims(tf.expand_dims(gamma, -1), -1) * z_mu_outer, 0)
               / tf.expand_dims(tf.expand_dims(sum_gamma, -1), -1))

        return phi, mu, cov

    def gmm_energy(self,
                   z: tf.Tensor,
                   phi: tf.Tensor,
                   mu: tf.Tensor,
                   cov: tf.Tensor,
                   return_mean: bool = True) \
            -> Tuple[tf.Tensor, tf.Tensor]:

        D = tf.shape(cov)[1]

        # N x K x D
        z_mu = tf.expand_dims(z, 1) - tf.expand_dims(mu, 0)

        # cholesky decomposition of covariance
        # K x D x D
        eps = 1e-6
        L = tf.linalg.cholesky(cov + tf.eye(D) * eps)

        # K
        # log(det(cov)) = 2 * sum[log(diag(L))] -> OK
        log_det_cov = 2. * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), 1)

        # K x D x N
        z_mu_T = tf.transpose(z_mu, perm=[1, 2, 0])

        # K x D x D
        v = tf.linalg.triangular_solve(L, z_mu_T, lower=True)

        # K x N
        logits = tf.math.log(tf.expand_dims(phi, -1)) - .5 * (
                tf.reduce_sum(tf.square(v), 1)
                + tf.cast(D, tf.float32) * tf.math.log(2. * np.pi)
                + tf.expand_dims(log_det_cov, -1))

        # N
        sample_energy = - tf.reduce_logsumexp(logits, axis=0)

        if return_mean:
            # 1
            sample_energy = tf.reduce_mean(sample_energy)  # 1

        # 1
        cov_diag = tf.reduce_sum(tf.divide(1, tf.linalg.diag_part(cov)))

        return sample_energy, cov_diag

    def loss_fn(self,
                x: tf.Tensor,
                recon_x: tf.Tensor,
                z: tf.Tensor,
                gamma: tf.Tensor,
                w_energy: float = .1,
                w_cov_diag: float = .005) \
            -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        # TODO: loss fn should be added to model like VAE?!
        recon_error = tf.reduce_mean((x - recon_x) ** 2)
        phi, mu, cov = self.gmm_params(z, gamma)
        sample_energy, cov_diag = self.gmm_energy(z, phi, mu, cov)
        loss = recon_error + w_energy * sample_energy + w_cov_diag * cov_diag
        return loss, sample_energy, recon_error, cov_diag
