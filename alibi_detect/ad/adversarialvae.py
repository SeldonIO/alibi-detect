import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import kld
import tensorflow_probability as tfp
from typing import Dict, Tuple
from alibi_detect.models.autoencoder import VAE
from alibi_detect.models.trainer import trainer
from alibi_detect.models.losses import loss_adv_vae
from alibi_detect.base import BaseDetector, FitMixin, ThresholdMixin, adversarial_prediction_dict

logger = logging.getLogger(__name__)


class AdversarialVAE(BaseDetector, FitMixin, ThresholdMixin):

    def __init__(self,
                 threshold: float = None,
                 vae: tf.keras.Model = None,
                 model: tf.keras.Model = None,
                 encoder_net: tf.keras.Sequential = None,
                 decoder_net: tf.keras.Sequential = None,
                 latent_dim: int = None,
                 samples: int = 10,
                 beta: float = 0.,
                 data_type: str = None
                 ) -> None:
        """
        VAE-based adversarial detector.

        Parameters
        ----------
        threshold
            Threshold used for adversarial score to determine adversarial instances.
        vae
            A trained tf.keras model if available.
        model
            A trained tf.keras classification model.
        encoder_net
            Layers for the encoder wrapped in a tf.keras.Sequential class if no 'vae' is specified.
        decoder_net
            Layers for the decoder wrapped in a tf.keras.Sequential class if no 'vae' is specified.
        latent_dim
            Dimensionality of the latent space.
        samples
            Number of samples sampled to evaluate each instance.
        beta
            Beta parameter for KL-divergence loss term.
        data_type
            Optionally specifiy the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()

        if threshold is None:
            logger.warning('No threshold level set. Need to infer threshold using `infer_threshold`.')

        self.threshold = threshold
        self.samples = samples
        self.model = model
        for layer in self.model.layers:  # freeze model layers
            layer.trainable = False

        # check if model can be loaded, otherwise initialize VAE model
        if isinstance(vae, tf.keras.Model):
            self.vae = vae
        elif isinstance(encoder_net, tf.keras.Sequential) and isinstance(decoder_net, tf.keras.Sequential):
            self.vae = VAE(encoder_net, decoder_net, latent_dim, beta=beta)  # define VAE model
        else:
            raise TypeError('No valid format detected for `vae` (tf.keras.Model) '
                            'or `encoder_net` and `decoder_net` (tf.keras.Sequential).')

        # set metadata
        self.meta['detector_type'] = 'offline'
        self.meta['data_type'] = data_type

    def fit(self,
            X: np.ndarray,
            loss_fn: tf.keras.losses = loss_adv_vae,
            w_model: float = 1.,
            w_recon: float = 0.,
            optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam(learning_rate=1e-3),
            cov_elbo: dict = None,
            epochs: int = 20,
            batch_size: int = 64,
            verbose: bool = True,
            log_metric: Tuple[str, "tf.keras.metrics"] = None,
            callbacks: tf.keras.callbacks = None,
            ) -> None:
        """
        Train Adversarial VAE model.

        Parameters
        ----------
        X
            Training batch.
        loss_fn
            Loss function used for training.
        w_model
            Weight on model prediction loss term.
        w_recon
            Weight on elbo loss term.
        optimizer
            Optimizer used for training.
        cov_elbo
            Dictionary with covariance matrix options in case the elbo loss function is used.
            Either use the full covariance matrix inferred from X (dict(cov_full=None)),
            only the variance (dict(cov_diag=None)) or a float representing the same standard deviation
            for each feature (e.g. dict(sim=.05)).
        epochs
            Number of training epochs.
        batch_size
            Batch size used for training.
        verbose
            Whether to print training progress.
        log_metric
            Additional metrics whose progress will be displayed if verbose equals True.
        callbacks
            Callbacks used during training.
        """
        # train arguments
        args = [self.vae, loss_fn, X]
        kwargs = {'optimizer': optimizer,
                  'epochs': epochs,
                  'batch_size': batch_size,
                  'verbose': verbose,
                  'log_metric': log_metric,
                  'callbacks': callbacks,
                  'loss_fn_kwargs': {'w_model': w_model,
                                     'w_recon': w_recon,
                                     'model': self.model}
                  }

        # initialize covariance matrix if default adversarial vae loss is used with elbo enabled
        use_elbo = loss_fn.__name__ == 'loss_adv_vae' and cov_elbo
        if use_elbo:
            cov_elbo_type, cov = [*cov_elbo][0], [*cov_elbo.values()][0]
            if cov_elbo_type in ['cov_full', 'cov_diag']:
                cov = tfp.stats.covariance(X.reshape(X.shape[0], -1))
                if cov_elbo_type == 'cov_diag':  # infer standard deviation from covariance matrix
                    cov = tf.math.sqrt(tf.linalg.diag_part(cov))
            kwargs['loss_fn_kwargs'][cov_elbo_type] = tf.dtypes.cast(cov, tf.float32)

        # train
        trainer(*args, **kwargs)

    def infer_threshold(self,
                        X: np.ndarray,
                        threshold_perc: float = 95.
                        ) -> None:
        """
        Update threshold by a value inferred from the percentage of instances considered to be
        adversarial in a sample of the dataset.

        Parameters
        ----------
        X
            Batch of instances.
        threshold_perc
            Percentage of X considered to be normal based on the adversarial score.
        """
        # compute adversarial scores
        adv_score = self.score(X)

        # update threshold
        self.threshold = np.percentile(adv_score, threshold_perc)

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute adversarial scores.

        Parameters
        ----------
        X
            Batch of instances to analyze.

        Returns
        -------
        Array with adversarial scores for each instance in the batch.
        """
        # sample reconstructed instances
        X_samples = np.repeat(X, self.samples, axis=0)
        X_recon = self.vae(X_samples)

        # model predictions
        y = self.model(X_samples)
        y_recon = self.model(X_recon)

        # KL-divergence between predictions
        kld_y = kld(y, y_recon).numpy().reshape(-1, self.samples)
        adv_score = np.mean(kld_y, axis=1)
        return adv_score

    def predict(self,
                X: np.ndarray,
                return_instance_score: bool = True) \
            -> Dict[Dict[str, str], Dict[np.ndarray, np.ndarray]]:
        """
        Predict whether instances are adversarial instances or not.

        Parameters
        ----------
        X
            Batch of instances.
        return_instance_score
            Whether to return instance level adversarial scores.

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the adversarial predictions and instance level adversarial scores.
        """
        adv_score = self.score(X)

        # values above threshold are outliers
        adv_pred = (adv_score > self.threshold).astype(int)

        # populate output dict
        ad = adversarial_prediction_dict()
        ad['meta'] = self.meta
        ad['data']['is_adversarial'] = adv_pred
        if return_instance_score:
            ad['data']['instance_score'] = adv_score
        return ad
