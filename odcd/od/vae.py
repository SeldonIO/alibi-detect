import logging
import numpy as np
import tensorflow as tf
from typing import Tuple
from odcd.models.autoencoder import VAE
from odcd.models.trainer import trainer
from odcd.models.losses import elbo

logger = logging.getLogger(__name__)

# TODO: for tabular data -> highlight anomalous features, for images -> patches and ts -> windows; mse and proba space
# TODO: reconstruction proba; make sure to infer correct distribution


class OutlierVAE:

    def __init__(self,
                 threshold: float,
                 score_type: str,  # mse, cross-entropy, proba (or combo)
                 encoder_net: tf.keras.Sequential,
                 decoder_net: tf.keras.Sequential,
                 latent_dim: int,
                 loss_fn: tf.keras.losses = elbo,
                 optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam(learning_rate=1e-3),
                 epochs: int = 20,
                 batch_size: int = 64,
                 verbose: bool = True,
                 log_metric: Tuple[str, tf.keras.metrics] = None,
                 callbacks: tf.keras.callbacks = None,
                 samples: int = 10
                 ) -> None:

        self.threshold = threshold  # allow for threshold of both proba and/or mse/cross-entropy
        self.score_type = score_type
        self.samples = samples
        self.vae = VAE(encoder_net, decoder_net, latent_dim)  # define VAE model
        self.train_args = [self.vae, loss_fn]
        self.train_kwargs = {'optimizer': optimizer,
                             'epochs': epochs,
                             'batch_size': batch_size,
                             'verbose': verbose,
                             'log_metric': log_metric,
                             'callbacks': callbacks}

    def fit(self, X: np.ndarray) -> None:
        args = self.train_args + [X]
        trainer(*args, **self.train_kwargs)

    def score(self, X: np.ndarray) -> np.ndarray:

        shape = X.shape
        axes = tuple(np.arange(len(shape)))

        # sample reconstructed instances
        X_samples = np.repeat(X, self.samples, axis=0)
        X_recon = self.vae(X_samples)

        if self.score_type in ['mse', 'mse-proba']:
            # calculate mean reconstruction errors
            se = np.power(X_samples - X_recon, 2)
            mse = np.mean(se, axis=axes[1:])
            # convert into outlier score
            outlier_score = mse.reshape(shape[0], -1).mean(axis=1)
        else:
            pass
        return outlier_score

    def predict(self, X: np.ndarray) -> np.ndarray:
        outlier_score = self.score(X)  # compute outlier scores
        outlier_pred = (outlier_score > self.threshold).astype(int)  # convert outlier scores into predictions
        return outlier_pred
