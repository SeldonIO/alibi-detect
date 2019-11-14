import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Layer
from typing import Callable, Tuple
from alibi_detect.utils.distance import relative_euclidean_distance

# TODO: add difference between train and inference mode for dropout


class Sampling(Layer):
    """ Reparametrization trick. Uses (z_mean, z_log_var) to sample the latent vector z. """

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Sample z.

        Parameters
        ----------
        inputs
            Tuple with mean and log variance.

        Returns
        -------
        Sampled vector z.
        """
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class EncoderVAE(Layer):

    def __init__(self,
                 encoder_net: tf.keras.Sequential,
                 latent_dim: int,
                 name: str = 'encoder_vae') -> None:
        """
        Encoder of VAE.

        Parameters
        ----------
        encoder_net
            Layers for the encoder wrapped in a tf.keras.Sequential class.
        latent_dim
            Dimensionality of the latent space.
        name
            Name of encoder.
        """
        super(EncoderVAE, self).__init__(name=name)
        self.encoder_net = encoder_net
        self.fc_mean = Dense(latent_dim, activation=None)
        self.fc_log_var = Dense(latent_dim, activation=tf.nn.softplus)
        self.sampling = Sampling()

    def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        x = self.encoder_net(x)
        if len(x.shape) > 2:
            x = Flatten()(x)
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(Layer):

    def __init__(self,
                 decoder_net: tf.keras.Sequential,
                 name: str = 'decoder') -> None:
        """
        Decoder of (V)AE.

        Parameters
        ----------
        decoder_net
            Layers for the decoder wrapped in a tf.keras.Sequential class.
        name
            Name of decoder.
        """
        super(Decoder, self).__init__(name=name)
        self.decoder_net = decoder_net

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.decoder_net(x)


class VAE(tf.keras.Model):

    def __init__(self,
                 encoder_net: tf.keras.Sequential,
                 decoder_net: tf.keras.Sequential,
                 latent_dim: int,
                 beta: float = 1.,
                 name: str = 'vae') -> None:
        """
        Combine encoder and decoder in VAE.

        Parameters
        ----------
        encoder_net
            Layers for the encoder wrapped in a tf.keras.Sequential class.
        decoder_net
            Layers for the decoder wrapped in a tf.keras.Sequential class.
        latent_dim
            Dimensionality of the latent space.
        beta
            Beta parameter for KL-divergence loss term.
        name
            Name of VAE model.
        """
        super(VAE, self).__init__(name=name)
        self.encoder = EncoderVAE(encoder_net, latent_dim)
        self.decoder = Decoder(decoder_net)
        self.beta = beta
        self.latent_dim = latent_dim

    def call(self, x: tf.Tensor) -> tf.Tensor:
        z_mean, z_log_var, z = self.encoder(x)
        x_recon = self.decoder(z)
        # add KL divergence loss term
        kl_loss = -.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(self.beta * kl_loss)
        return x_recon


class EncoderAE(Layer):

    def __init__(self,
                 encoder_net: tf.keras.Sequential,
                 name: str = 'encoder_ae') -> None:
        """
        Encoder of AE.

        Parameters
        ----------
        encoder_net
            Layers for the encoder wrapped in a tf.keras.Sequential class.
        name
            Name of encoder.
        """
        super(EncoderAE, self).__init__(name=name)
        self.encoder_net = encoder_net

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.encoder_net(x)


class AE(tf.keras.Model):

    def __init__(self,
                 encoder_net: tf.keras.Sequential,
                 decoder_net: tf.keras.Sequential,
                 name: str = 'ae') -> None:
        """
        Combine encoder and decoder in AE.

        Parameters
        ----------
        encoder_net
            Layers for the encoder wrapped in a tf.keras.Sequential class.
        decoder_net
            Layers for the decoder wrapped in a tf.keras.Sequential class.
        name
            Name of autoencoder model.
        """
        super(AE, self).__init__(name=name)
        self.encoder = EncoderAE(encoder_net)
        self.decoder = Decoder(decoder_net)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon


def eucl_cosim_features(x: tf.Tensor,
                        y: tf.Tensor,
                        max_eucl: float = 1e2) -> tf.Tensor:
    """
    Compute features extracted from the reconstructed instance using the
    relative Euclidean distance and cosine similarity between 2 tensors.

    Parameters
    ----------
    x
        Tensor used in feature computation.
    y
        Tensor used in feature computation.
    max_eucl
        Maximum value to clip relative Euclidean distance by.

    Returns
    -------
    Tensor concatenating the relative Euclidean distance and
    cosine similarity features.
    """
    if len(x.shape) > 2 or len(y.shape) > 2:
        x = Flatten()(x)
        y = Flatten()(y)
    rec_cos = tf.reshape(tf.keras.losses.cosine_similarity(y, x, -1), (-1, 1))
    rec_euc = tf.reshape(relative_euclidean_distance(y, x, -1), (-1, 1))
    # rec_euc could become very large so should be clipped
    rec_euc = tf.clip_by_value(rec_euc, 0, max_eucl)
    return tf.concat([rec_cos, rec_euc], -1)


class AEGMM(tf.keras.Model):

    def __init__(self,
                 encoder_net: tf.keras.Sequential,
                 decoder_net: tf.keras.Sequential,
                 gmm_density_net: tf.keras.Sequential,
                 n_gmm: int,
                 recon_features: Callable = eucl_cosim_features,
                 name: str = 'aegmm') -> None:
        """
        Deep Autoencoding Gaussian Mixture Model.

        Parameters
        ----------
        encoder_net
            Layers for the encoder wrapped in a tf.keras.Sequential class.
        decoder_net
            Layers for the decoder wrapped in a tf.keras.Sequential class.
        gmm_density_net
            Layers for the GMM network wrapped in a tf.keras.Sequential class.
        n_gmm
            Number of components in GMM.
        recon_features
            Function to extract features from the reconstructed instance by the decoder.
        name
            Name of the AEGMM model.
        """
        super(AEGMM, self).__init__(name=name)
        self.encoder = encoder_net
        self.decoder = decoder_net
        self.gmm_density = gmm_density_net
        self.n_gmm = n_gmm
        self.recon_features = recon_features

    def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        enc = self.encoder(x)
        x_recon = self.decoder(enc)
        recon_features = self.recon_features(x, x_recon)
        z = tf.concat([enc, recon_features], -1)
        gamma = self.gmm_density(z)
        return x_recon, z, gamma


class VAEGMM(tf.keras.Model):

    def __init__(self,
                 encoder_net: tf.keras.Sequential,
                 decoder_net: tf.keras.Sequential,
                 gmm_density_net: tf.keras.Sequential,
                 n_gmm: int,
                 latent_dim: int,
                 recon_features: Callable = eucl_cosim_features,
                 beta: float = 1.,
                 name: str = 'vaegmm') -> None:
        """
        Variational Autoencoding Gaussian Mixture Model.

        Parameters
        ----------
        encoder_net
            Layers for the encoder wrapped in a tf.keras.Sequential class.
        decoder_net
            Layers for the decoder wrapped in a tf.keras.Sequential class.
        gmm_density_net
            Layers for the GMM network wrapped in a tf.keras.Sequential class.
        n_gmm
            Number of components in GMM.
        latent_dim
            Dimensionality of the latent space.
        recon_features
            Function to extract features from the reconstructed instance by the decoder.
        beta
            Beta parameter for KL-divergence loss term.
        name
            Name of the VAEGMM model.
        """
        super(VAEGMM, self).__init__(name=name)
        self.encoder = EncoderVAE(encoder_net, latent_dim)
        self.decoder = decoder_net
        self.gmm_density = gmm_density_net
        self.n_gmm = n_gmm
        self.latent_dim = latent_dim
        self.recon_features = recon_features
        self.beta = beta

    def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        enc_mean, enc_log_var, enc = self.encoder(x)
        x_recon = self.decoder(enc)
        recon_features = self.recon_features(x, x_recon)
        z = tf.concat([enc, recon_features], -1)
        gamma = self.gmm_density(z)
        # add KL divergence loss term
        kl_loss = -.5 * tf.reduce_mean(enc_log_var - tf.square(enc_mean) - tf.exp(enc_log_var) + 1)
        self.add_loss(self.beta * kl_loss)
        return x_recon, z, gamma
