import logging
import numpy as np
import tensorflow as tf
from typing import Dict, Tuple
from odcd.models.autoencoder import VAE
from odcd.models.trainer import trainer
from odcd.models.losses import elbo
from odcd.od.base import BaseOutlierDetector, FitMixin, ThresholdMixin, outlier_prediction_dict

logger = logging.getLogger(__name__)


class OutlierVAE(BaseOutlierDetector, FitMixin, ThresholdMixin):

    def __init__(self,
                 threshold: float = None,
                 score_type: str = 'mse',  # TODO: reconstruction proba; make sure to infer correct distribution
                 vae: tf.keras.Model = None,
                 encoder_net: tf.keras.Sequential = None,
                 decoder_net: tf.keras.Sequential = None,
                 latent_dim: int = None,
                 samples: int = 10,
                 beta: float = 1.,
                 data_type: str = None
                 ) -> None:
        """
        VAE-based outlier detector.

        Parameters
        ----------
        threshold
            Threshold used for outlier score to determine outliers.
        score_type
            Metric used for outlier scores. Either 'mse' (mean squared error) or
            'proba' (reconstruction probabilities) supported.
        vae
            A trained tf.keras model if available.
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
        self.score_type = score_type
        self.samples = samples

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
            loss_fn: tf.keras.losses = elbo,
            optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam(learning_rate=1e-3),
            epochs: int = 20,
            batch_size: int = 64,
            verbose: bool = True,
            log_metric: Tuple[str, "tf.keras.metrics"] = None,
            callbacks: tf.keras.callbacks = None,
            ) -> None:
        """
        Train VAE model.

        Parameters
        ----------
        X
            Training batch.
        loss_fn
            Loss function used for training.
        optimizer
            Optimizer used for training.
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
                  'callbacks': callbacks}
        trainer(*args, **kwargs)

    def infer_threshold(self, X: np.ndarray, *args, **kwargs) -> None:
        # TODO infer threshold level from e.g. percentile
        pass

    def feature_score(self, X_orig: np.ndarray, X_recon: np.ndarray) -> np.ndarray:
        """
        Compute feature level outlier scores.

        Parameters
        ----------
        X_orig
            Batch of original instances.
        X_recon
            Batch of reconstructed instances.

        Returns
        -------
        Feature level outlier scores.
        """
        if self.score_type == 'mse':
            fscore = np.power(X_orig - X_recon, 2)
            fscore = fscore.reshape((-1, self.samples) + X_orig.shape[1:])
            fscore = np.mean(fscore, axis=1)
        elif self.score_type == 'proba':
            pass
        return fscore

    def instance_score(self, fscore: np.ndarray) -> np.ndarray:
        """
        Compute instance level outlier scores.

        Parameters
        ----------
        fscore
            Feature level outlier scores.

        Returns
        -------
        Instance level outlier scores.
        """
        axes = tuple(range(len(fscore.shape)))
        return np.mean(fscore, axis=axes[1:])

    def score(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute feature and instance level outlier scores.

        Parameters
        ----------
        X
            Batch of instances.

        Returns
        -------
        Feature and instance level outlier scores.
        """
        # sample reconstructed instances
        X_samples = np.repeat(X, self.samples, axis=0)
        X_recon = self.vae(X_samples)

        # compute feature and instance level scores
        fscore = self.feature_score(X_samples, X_recon)
        iscore = self.instance_score(fscore)

        return fscore, iscore

    def predict(self,
                X: np.ndarray,
                outlier_type: str = 'instance',
                return_feature_score: bool = True,
                return_instance_score: bool = True) \
            -> Dict[Dict[str, str], Dict[np.ndarray, np.ndarray]]:
        """
        Predict whether instances are outliers or not.

        Parameters
        ----------
        X
            Batch of instances.
        outlier_type
            Predict outliers at the 'feature' or 'instance' level.
        return_feature_score
            Whether to return feature level outlier scores.
        return_instance_score
            Whether to return instance level outlier scores.

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the outlier predictions and both feature and instance level outlier scores.
        """

        # compute outlier scores
        fscore, iscore = self.score(X)
        if outlier_type == 'feature':
            outlier_score = fscore
        elif outlier_type == 'instance':
            outlier_score = iscore
        else:
            raise ValueError('`outlier_score` needs to be either `feature` or `instance`.')

        # values above threshold are outliers
        outlier_pred = (outlier_score > self.threshold).astype(int)

        # populate output dict
        od = outlier_prediction_dict()
        od['meta'] = self.meta
        od['data']['is_outlier'] = outlier_pred
        if return_feature_score:
            od['data']['feature_score'] = fscore
        if return_instance_score:
            od['data']['instance_score'] = iscore
        return od
