import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, Concatenate, Dense, Flatten, Layer, LSTM
from typing import Callable, List, Tuple
from alibi_detect.utils.tensorflow.distance import relative_euclidean_distance


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
        self.fc_log_var = Dense(latent_dim, activation=None)
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


class EncoderLSTM(Layer):

    def __init__(self,
                 latent_dim: int,
                 name: str = 'encoder_lstm') -> None:
        """
        Bidirectional LSTM encoder.

        Parameters
        ----------
        latent_dim
            Latent dimension. Must be an even number given the bidirectional encoder.
        name
            Name of encoder.
        """
        super(EncoderLSTM, self).__init__(name=name)
        self.encoder_net = Bidirectional(LSTM(latent_dim // 2, return_state=True, return_sequences=True))

    def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        enc_out, fwd_h, fwd_c, bwd_h, bwd_c = self.encoder_net(x)
        h = Concatenate()([fwd_h, bwd_h])
        c = Concatenate()([fwd_c, bwd_c])
        return enc_out, [h, c]


class DecoderLSTM(Layer):

    def __init__(self,
                 latent_dim: int,
                 output_dim: int,
                 output_activation: str = None,
                 name: str = 'decoder_lstm') -> None:
        """
        LSTM decoder.

        Parameters
        ----------
        latent_dim
            Latent dimension.
        output_dim
            Decoder output dimension.
        output_activation
            Activation used in the Dense output layer.
        name
            Name of decoder.
        """
        super(DecoderLSTM, self).__init__(name=name)
        self.decoder_net = LSTM(latent_dim, return_state=True, return_sequences=True)
        self.dense = Dense(output_dim, activation=output_activation)

    def call(self, x: tf.Tensor, init_state: List[tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor, List[tf.Tensor]]:
        x, h, c = self.decoder_net(x, initial_state=init_state)
        dec_out = self.dense(x)
        return dec_out, x, [h, c]


class Seq2Seq(tf.keras.Model):

    def __init__(self,
                 encoder_net: EncoderLSTM,
                 decoder_net: DecoderLSTM,
                 threshold_net: tf.keras.Sequential,
                 n_features: int,
                 score_fn: Callable = tf.math.squared_difference,
                 beta: float = 1.,
                 name: str = 'seq2seq') -> None:
        """
        Sequence-to-sequence model.

        Parameters
        ----------
        encoder_net
            Encoder network.
        decoder_net
            Decoder network.
        threshold_net
            Regression network used to estimate threshold.
        n_features
            Number of features.
        score_fn
            Function used for outlier score.
        beta
            Weight on the threshold estimation loss term.
        name
            Name of the seq2seq model.
        """
        super(Seq2Seq, self).__init__(name=name)
        self.encoder = encoder_net
        self.decoder = decoder_net
        self.threshold_net = threshold_net
        self.threshold_est = Dense(n_features, activation=None)
        self.score_fn = score_fn
        self.beta = beta

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """ Forward pass used for teacher-forcing training. """
        # reconstruct input via encoder-decoder
        init_state = self.encoder(x)[1]
        x_recon, z, _ = self.decoder(x, init_state=init_state)

        # compute outlier score
        err_recon = self.score_fn(x, x_recon)

        # estimate outlier threshold from hidden state of decoder
        z = self.threshold_net(z)
        threshold_est = self.threshold_est(z)

        # add threshold estimate loss
        threshold_loss = tf.reduce_mean((err_recon - threshold_est) ** 2)
        self.add_loss(self.beta * threshold_loss)
        return x_recon

    def decode_seq(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Sequence decoding and threshold estimation used for inference. """
        seq_len = np.shape(x)[1]
        n_batch = x.shape[0]

        # use encoder to get state vectors
        init_state = self.encoder(x)[1]

        # generate start of target sequence
        decoder_input = np.reshape(x[:, 0, :], (n_batch, 1, -1))

        # initialize hidden states used to compute outlier thresholds
        z = np.zeros((n_batch, seq_len, init_state[0].numpy().shape[1])).astype(np.float32)

        # sequential prediction of time series
        decoded_seq = np.zeros_like(x)
        decoded_seq[:, 0, :] = x[:, 0, :]
        i = 1
        while i < seq_len:
            # decode step in sequence
            decoder_output = self.decoder(decoder_input, init_state=init_state)
            decoded_seq[:, i:i+1, :] = decoder_output[0].numpy()
            init_state = decoder_output[2]

            # update hidden state decoder used for outlier threshold
            z[:, i:i+1, :] = decoder_output[1].numpy()

            # update next decoder input
            decoder_input = np.zeros_like(decoder_input)
            decoder_input[:, :1, :] = decoder_output[0].numpy()

            i += 1

        # compute outlier thresholds
        z = self.threshold_net(z)
        threshold_est = self.threshold_est(z).numpy()

        return decoded_seq, threshold_est


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
