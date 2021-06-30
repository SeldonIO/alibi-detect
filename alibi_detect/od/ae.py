import logging
import numpy as np
import tensorflow as tf
from typing import Dict, Tuple
from alibi_detect.models.tensorflow.autoencoder import AE
from alibi_detect.models.tensorflow.trainer import trainer
from alibi_detect.base import BaseDetector, FitMixin, ThresholdMixin, outlier_prediction_dict
from alibi_detect.utils.tensorflow.prediction import predict_batch

logger = logging.getLogger(__name__)


class OutlierAE(BaseDetector, FitMixin, ThresholdMixin):

    def __init__(self,
                 threshold: float = None,
                 ae: tf.keras.Model = None,
                 encoder_net: tf.keras.Sequential = None,
                 decoder_net: tf.keras.Sequential = None,
                 data_type: str = None
                 ) -> None:
        """
        AE-based outlier detector.

        Parameters
        ----------
        threshold
            Threshold used for outlier score to determine outliers.
        ae
            A trained tf.keras model if available.
        encoder_net
            Layers for the encoder wrapped in a tf.keras.Sequential class if no 'ae' is specified.
        decoder_net
            Layers for the decoder wrapped in a tf.keras.Sequential class if no 'ae' is specified.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()

        if threshold is None:
            logger.warning('No threshold level set. Need to infer threshold using `infer_threshold`.')

        self.threshold = threshold

        # check if model can be loaded, otherwise initialize AE model
        if isinstance(ae, tf.keras.Model):
            self.ae = ae
        elif isinstance(encoder_net, tf.keras.Sequential) and isinstance(decoder_net, tf.keras.Sequential):
            self.ae = AE(encoder_net, decoder_net)
        else:
            raise TypeError('No valid format detected for `ae` (tf.keras.Model) '
                            'or `encoder_net`, `decoder_net` (tf.keras.Sequential).')

        # set metadata
        self.meta['detector_type'] = 'offline'
        self.meta['data_type'] = data_type

    def fit(self,
            X: np.ndarray,
            loss_fn: tf.keras.losses = tf.keras.losses.MeanSquaredError(),
            optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam(learning_rate=1e-3),
            epochs: int = 20,
            batch_size: int = 64,
            verbose: bool = True,
            log_metric: Tuple[str, "tf.keras.metrics"] = None,
            callbacks: tf.keras.callbacks = None,
            ) -> None:
        """
        Train AE model.

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
        args = [self.ae, loss_fn, X]
        kwargs = {'optimizer': optimizer,
                  'epochs': epochs,
                  'batch_size': batch_size,
                  'verbose': verbose,
                  'log_metric': log_metric,
                  'callbacks': callbacks}

        # train
        trainer(*args, **kwargs)

    def infer_threshold(self,
                        X: np.ndarray,
                        outlier_type: str = 'instance',
                        outlier_perc: float = 100.,
                        threshold_perc: float = 95.,
                        batch_size: int = int(1e10)
                        ) -> None:
        """
        Update threshold by a value inferred from the percentage of instances considered to be
        outliers in a sample of the dataset.

        Parameters
        ----------
        X
            Batch of instances.
        outlier_type
            Predict outliers at the 'feature' or 'instance' level.
        outlier_perc
            Percentage of sorted feature level outlier scores used to predict instance level outlier.
        threshold_perc
            Percentage of X considered to be normal based on the outlier score.
        batch_size
            Batch size used when making predictions with the autoencoder.
        """
        # compute outlier scores
        fscore, iscore = self.score(X, outlier_perc=outlier_perc, batch_size=batch_size)
        if outlier_type == 'feature':
            outlier_score = fscore
        elif outlier_type == 'instance':
            outlier_score = iscore
        else:
            raise ValueError('`outlier_score` needs to be either `feature` or `instance`.')

        # update threshold
        self.threshold = np.percentile(outlier_score, threshold_perc)

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
        fscore = np.power(X_orig - X_recon, 2)
        return fscore

    def instance_score(self, fscore: np.ndarray, outlier_perc: float = 100.) -> np.ndarray:
        """
        Compute instance level outlier scores.

        Parameters
        ----------
        fscore
            Feature level outlier scores.
        outlier_perc
            Percentage of sorted feature level outlier scores used to predict instance level outlier.

        Returns
        -------
        Instance level outlier scores.
        """
        fscore_flat = fscore.reshape(fscore.shape[0], -1).copy()
        n_score_features = int(np.ceil(.01 * outlier_perc * fscore_flat.shape[1]))
        sorted_fscore = np.sort(fscore_flat, axis=1)
        sorted_fscore_perc = sorted_fscore[:, -n_score_features:]
        iscore = np.mean(sorted_fscore_perc, axis=1)
        return iscore

    def score(self, X: np.ndarray, outlier_perc: float = 100., batch_size: int = int(1e10)) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute feature and instance level outlier scores.

        Parameters
        ----------
        X
            Batch of instances.
        outlier_perc
            Percentage of sorted feature level outlier scores used to predict instance level outlier.
        batch_size
            Batch size used when making predictions with the autoencoder.

        Returns
        -------
        Feature and instance level outlier scores.
        """
        # reconstruct instances
        X_recon = predict_batch(X, self.ae, batch_size=batch_size)

        # compute feature and instance level scores
        fscore = self.feature_score(X, X_recon)
        iscore = self.instance_score(fscore, outlier_perc=outlier_perc)

        return fscore, iscore

    def predict(self,
                X: np.ndarray,
                outlier_type: str = 'instance',
                outlier_perc: float = 100.,
                batch_size: int = int(1e10),
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
        outlier_perc
            Percentage of sorted feature level outlier scores used to predict instance level outlier.
        batch_size
            Batch size used when making predictions with the autoencoder.
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
        fscore, iscore = self.score(X, outlier_perc=outlier_perc, batch_size=batch_size)
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
