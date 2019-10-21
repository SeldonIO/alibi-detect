import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Layer
from typing import Tuple

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
    """ VAE encoder. """

    def __init__(self,
                 encoder_net: tf.keras.Sequential,
                 latent_dim: int,
                 name: str = 'encoder_vae') -> None:
        super(EncoderVAE, self).__init__(name=name)
        self.encoder_net = encoder_net
        self.fc_mean = Dense(latent_dim, activation=None)
        self.fc_log_var = Dense(latent_dim, activation=tf.nn.softplus)
        self.sampling = Sampling()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.encoder_net(x)
        if len(x.shape) > 2:
            x = Flatten()(x)
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(Layer):
    """ (V)AE decoder. """

    def __init__(self,
                 decoder_net: tf.keras.Sequential,
                 name: str = 'decoder') -> None:
        super(Decoder, self).__init__(name=name)
        self.decoder_net = decoder_net

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.decoder_net(x)


class VAE(tf.keras.Model):
    """ Combine encoder and decoder in VAE. """

    def __init__(self,
                 encoder_net: tf.keras.Sequential,
                 decoder_net: tf.keras.Sequential,
                 latent_dim: int,
                 beta: float = 1.,
                 name: str = 'vae') -> None:
        super(VAE, self).__init__(name=name)
        self.encoder = EncoderVAE(encoder_net, latent_dim)
        self.decoder = Decoder(decoder_net)
        self.beta = beta

    def call(self, x: tf.Tensor) -> tf.Tensor:
        z_mean, z_log_var, z = self.encoder(x)
        x_recon = self.decoder(z)
        # add KL divergence loss term
        kl_loss = -.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(self.beta * kl_loss)
        return x_recon


class EncoderAE(Layer):
    """ Autoencoder (AE). """

    def __init__(self,
                 encoder_net: tf.keras.Sequential,
                 name: str = 'encoder_ae') -> None:
        super(EncoderAE, self).__init__(name=name)
        self.encoder_net = encoder_net

    def call(self, x: tf.Tensor):
        return self.encoder_net(x)


class AE(tf.keras.Model):
    """ Combine encoder and decoder in AE. """

    def __init__(self,
                 encoder_net: tf.keras.Sequential,
                 decoder_net: tf.keras.Sequential,
                 name: str = 'ae') -> None:
        super(AE, self).__init__(name=name)
        self.encoder = EncoderAE(encoder_net)
        self.decoder = Decoder(decoder_net)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon
