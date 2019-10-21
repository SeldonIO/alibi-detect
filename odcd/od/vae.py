import logging
import numpy as np
import tensorflow as tf
from typing import Dict, Tuple
from odcd.models.autoencoder import VAE
from odcd.models.trainer import trainer
from odcd.models.losses import elbo
from odcd.od.base import BaseOutlierDetector, FitMixin, ThresholdMixin, outlier_prediction_dict

logger = logging.getLogger(__name__)

# TODO: reconstruction proba; make sure to infer correct distribution


class OutlierVAE(BaseOutlierDetector, FitMixin, ThresholdMixin):

    def __init__(self,
                 threshold: float = None,
                 score_type: str = 'mse',
                 vae: tf.keras.Model = None,
                 encoder_net: tf.keras.Sequential = None,
                 decoder_net: tf.keras.Sequential = None,
                 latent_dim: int = None,
                 samples: int = 10,
                 data_type: str = 'tabular'
                 ) -> None:
        super().__init__()

        if threshold is None:
            logger.warning('No threshold level set. Need to infer threshold using `infer_threshold`.')

        self.threshold = threshold
        self.score_type = score_type  # mse, cross-entropy, proba (or combo?)
        self.samples = samples

        # check if model can be loaded, otherwise initialize VAE model
        if isinstance(vae, tf.keras.Model):
            self.vae = vae
        elif isinstance(encoder_net, tf.keras.Sequential) and isinstance(decoder_net, tf.keras.Sequential):
            self.vae = VAE(encoder_net, decoder_net, latent_dim)  # define VAE model
        else:
            raise TypeError('No valid format detected for `vae` (tf.keras.Model) '
                            'or `encoder_net` and `decoder_net` (tf.keras.Sequential).')

        # set metadata
        self.meta['detector_type'] = 'offline'  # TODO: not specific enough? e.g. 'vae'?
        self.meta['data_type'] = data_type  # TODO: nice way to infer data_type!

    def fit(self,
            X: np.ndarray,
            loss_fn: tf.keras.losses = elbo,
            optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam(learning_rate=1e-3),
            epochs: int = 20,
            batch_size: int = 64,
            verbose: bool = True,
            log_metric=None,  # TODO: Tuple[str, tf.keras.Metrics] = None,
            callbacks: tf.keras.callbacks = None,
            ) -> None:
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
        if self.score_type == 'mse':
            fscore = np.power(X_orig - X_recon, 2)
            fscore = fscore.reshape((-1, self.samples) + X_orig.shape[1:])
            fscore = np.mean(fscore, axis=1)
        elif self.score_type == 'proba':
            pass
        return fscore

    def instance_score(self, fscore: np.ndarray) -> np.ndarray:
        axes = tuple(range(len(fscore.shape)))
        return np.mean(fscore, axis=axes[1:])

    def score(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

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
