import logging
import numpy as np
import tensorflow as tf
from typing import Callable, Dict, Tuple
from alibi_detect.models.tensorflow.autoencoder import AEGMM, eucl_cosim_features
from alibi_detect.models.tensorflow.gmm import gmm_energy, gmm_params
from alibi_detect.models.tensorflow.losses import loss_aegmm
from alibi_detect.models.tensorflow.trainer import trainer
from alibi_detect.base import BaseDetector, FitMixin, ThresholdMixin, outlier_prediction_dict
from alibi_detect.utils.tensorflow.prediction import predict_batch

logger = logging.getLogger(__name__)


class OutlierAEGMM(BaseDetector, FitMixin, ThresholdMixin):

    def __init__(self,
                 threshold: float = None,
                 aegmm: tf.keras.Model = None,
                 encoder_net: tf.keras.Model = None,
                 decoder_net: tf.keras.Model = None,
                 gmm_density_net: tf.keras.Model = None,
                 n_gmm: int = None,
                 recon_features: Callable = eucl_cosim_features,
                 data_type: str = None
                 ) -> None:
        """
        AEGMM-based outlier detector.

        Parameters
        ----------
        threshold
            Threshold used for outlier score to determine outliers.
        aegmm
            A trained tf.keras model if available.
        encoder_net
            Layers for the encoder wrapped in a tf.keras.Sequential class if no 'aegmm' is specified.
        decoder_net
            Layers for the decoder wrapped in a tf.keras.Sequential class if no 'aegmm' is specified.
        gmm_density_net
            Layers for the GMM network wrapped in a tf.keras.Sequential class.
        n_gmm
            Number of components in GMM.
        recon_features
            Function to extract features from the reconstructed instance by the decoder.
        data_type
            Optionally specifiy the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()

        if threshold is None:
            logger.warning('No threshold level set. Need to infer threshold using `infer_threshold`.')

        self.threshold = threshold

        # check if model can be loaded, otherwise initialize AEGMM model
        if isinstance(aegmm, tf.keras.Model):
            self.aegmm = aegmm
        elif (isinstance(encoder_net, tf.keras.Sequential) and
              isinstance(decoder_net, tf.keras.Sequential) and
              isinstance(gmm_density_net, tf.keras.Sequential)):
            self.aegmm = AEGMM(encoder_net, decoder_net, gmm_density_net, n_gmm, recon_features)
        else:
            raise TypeError('No valid format detected for `aegmm` (tf.keras.Model) '
                            'or `encoder_net`, `decoder_net` and `gmm_density_net` (tf.keras.Sequential).')

        # set metadata
        self.meta['detector_type'] = 'offline'
        self.meta['data_type'] = data_type

        self.phi, self.mu, self.cov, self.L, self.log_det_cov = None, None, None, None, None

    def fit(self,
            X: np.ndarray,
            loss_fn: tf.keras.losses = loss_aegmm,
            w_energy: float = .1,
            w_cov_diag: float = .005,
            optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam(learning_rate=1e-4),
            epochs: int = 20,
            batch_size: int = 64,
            verbose: bool = True,
            log_metric: Tuple[str, "tf.keras.metrics"] = None,
            callbacks: tf.keras.callbacks = None,
            ) -> None:
        """
        Train AEGMM model.

        Parameters
        ----------
        X
            Training batch.
        loss_fn
            Loss function used for training.
        w_energy
            Weight on sample energy loss term if default `loss_aegmm` loss fn is used.
        w_cov_diag
            Weight on covariance regularizing loss term if default `loss_aegmm` loss fn is used.
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
        args = [self.aegmm, loss_fn, X]
        kwargs = {'optimizer': optimizer,
                  'epochs': epochs,
                  'batch_size': batch_size,
                  'verbose': verbose,
                  'log_metric': log_metric,
                  'callbacks': callbacks,
                  'loss_fn_kwargs': {'w_energy': w_energy,
                                     'w_cov_diag': w_cov_diag}
                  }

        # train
        trainer(*args, **kwargs)

        # set GMM parameters
        x_recon, z, gamma = self.aegmm(X)
        self.phi, self.mu, self.cov, self.L, self.log_det_cov = gmm_params(z, gamma)

    def infer_threshold(self,
                        X: np.ndarray,
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
        threshold_perc
            Percentage of X considered to be normal based on the outlier score.
        batch_size
            Batch size used when making predictions with the AEGMM.
        """
        # compute outlier scores
        iscore = self.score(X, batch_size=batch_size)

        # update threshold
        self.threshold = np.percentile(iscore, threshold_perc)

    def score(self, X: np.ndarray, batch_size: int = int(1e10)) -> np.ndarray:
        """
        Compute outlier scores.

        Parameters
        ----------
        X
            Batch of instances to analyze.
        batch_size
            Batch size used when making predictions with the AEGMM.

        Returns
        -------
        Array with outlier scores for each instance in the batch.
        """
        _, z, _ = predict_batch(X, self.aegmm, batch_size=batch_size)
        energy, _ = gmm_energy(z, self.phi, self.mu, self.cov, self.L, self.log_det_cov, return_mean=False)
        return energy.numpy()

    def predict(self,
                X: np.ndarray,
                batch_size: int = int(1e10),
                return_instance_score: bool = True) \
            -> Dict[Dict[str, str], Dict[np.ndarray, np.ndarray]]:
        """
        Compute outlier scores and transform into outlier predictions.

        Parameters
        ----------
        X
            Batch of instances.
        batch_size
            Batch size used when making predictions with the AEGMM.
        return_instance_score
            Whether to return instance level outlier scores.

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the outlier predictions and instance level outlier scores.
        """
        # compute outlier scores
        iscore = self.score(X, batch_size=batch_size)

        # values above threshold are outliers
        outlier_pred = (iscore > self.threshold).astype(int)

        # populate output dict
        od = outlier_prediction_dict()
        od['meta'] = self.meta
        od['data']['is_outlier'] = outlier_pred
        if return_instance_score:
            od['data']['instance_score'] = iscore
        return od
