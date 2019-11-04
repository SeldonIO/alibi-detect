import logging
import numpy as np
import tensorflow as tf
from typing import Callable, Dict, Tuple
from odcd.models.autoencoder import VAEGMM, eucl_cosim_features
from odcd.models.gmm import gmm_energy, gmm_params
from odcd.models.losses import elbo, loss_vaegmm
from odcd.models.trainer import trainer
from odcd.od.base import BaseOutlierDetector, FitMixin, ThresholdMixin, outlier_prediction_dict

logger = logging.getLogger(__name__)


class OutlierVAEGMM(BaseOutlierDetector, FitMixin, ThresholdMixin):

    def __init__(self,
                 threshold: float = None,
                 vaegmm: tf.keras.Model = None,
                 encoder_net: tf.keras.Sequential = None,
                 decoder_net: tf.keras.Sequential = None,
                 gmm_density_net: tf.keras.Sequential = None,
                 n_gmm: int = None,
                 latent_dim: int = None,
                 samples: int = 10,
                 beta: float = 1.,
                 recon_features: Callable = eucl_cosim_features,
                 data_type: str = None
                 ) -> None:
        """
        VAEGMM-based outlier detector.

        Parameters
        ----------
        threshold
            Threshold used for outlier score to determine outliers.
        vaegmm
            A trained tf.keras model if available.
        encoder_net
            Layers for the encoder wrapped in a tf.keras.Sequential class if no 'vaegmm' is specified.
        decoder_net
            Layers for the decoder wrapped in a tf.keras.Sequential class if no 'vaegmm' is specified.
        gmm_density_net
            Layers for the GMM network wrapped in a tf.keras.Sequential class.
        n_gmm
            Number of components in GMM.
        latent_dim
            Dimensionality of the latent space.
        samples
            Number of samples sampled to evaluate each instance.
        beta
            Beta parameter for KL-divergence loss term.
        recon_features
            Function to extract features from the reconstructed instance by the decoder.
        data_type
            Optionally specifiy the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()

        if threshold is None:
            logger.warning('No threshold level set. Need to infer threshold using `infer_threshold`.')

        self.threshold = threshold
        self.samples = samples

        # check if model can be loaded, otherwise initialize AEGMM model
        if isinstance(vaegmm, tf.keras.Model):
            self.vaegmm = vaegmm
        elif (isinstance(encoder_net, tf.keras.Sequential) and
              isinstance(decoder_net, tf.keras.Sequential) and
              isinstance(gmm_density_net, tf.keras.Sequential)):
            self.vaegmm = VAEGMM(encoder_net, decoder_net, gmm_density_net, n_gmm,
                                 latent_dim, recon_features=recon_features, beta=beta)
        else:
            raise TypeError('No valid format detected for `vaegmm` (tf.keras.Model) '
                            'or `encoder_net`, `decoder_net` and `gmm_density_net` (tf.keras.Sequential).')

        # set metadata
        self.meta['detector_type'] = 'offline'
        self.meta['data_type'] = data_type

    def fit(self,
            X: np.ndarray,
            loss_fn: tf.keras.losses = loss_vaegmm,
            w_recon: float = 1e-7,
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
        Train VAEGMM model.

        Parameters
        ----------
        X
            Training batch.
        loss_fn
            Loss function used for training.
        w_recon
            Weight on elbo loss term if default `loss_vaegmm`.
        w_energy
            Weight on sample energy loss term if default `loss_vaegmm` loss fn is used.
        w_cov_diag
            Weight on covariance regularizing loss term if default `loss_vaegmm` loss fn is used.
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
        args = [self.vaegmm, loss_fn, X]
        kwargs = {'optimizer': optimizer,
                  'epochs': epochs,
                  'batch_size': batch_size,
                  'verbose': verbose,
                  'log_metric': log_metric,
                  'callbacks': callbacks,
                  'loss_fn_kwargs': {'w_recon': w_recon,
                                     'w_energy': w_energy,
                                     'w_cov_diag': w_cov_diag}
                  }

        # train
        trainer(*args, **kwargs)

        # set GMM parameters
        x_recon, z, gamma = self.aegmm(X)
        self.phi, self.mu, self.cov, self.L, self.log_det_cov = gmm_params(z, gamma)

    def infer_threshold(self,
                        X: np.ndarray,
                        threshold_perc: float = 95.
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
        """
        # compute outlier scores
        iscore = self.score(X)

        # update threshold
        self.threshold = np.percentile(iscore, threshold_perc)

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute outlier scores.

        Parameters
        ----------
        X
            Batch of instances to analyze.

        Returns
        -------
        Array with outlier scores for each instance in the batch.
        """
        # need to sample
        pass

    def predict(self,
                X: np.ndarray,
                return_instance_score: bool = True) \
            -> Dict[Dict[str, str], Dict[np.ndarray, np.ndarray]]:
        """
        Compute outlier scores and transform into outlier predictions.

        Parameters
        ----------
        X
            Batch of instances.
        return_instance_score
            Whether to return instance level outlier scores.

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the outlier predictions and both feature and instance level outlier scores.
        """
        # compute outlier scores
        iscore = self.score(X)

        # values above threshold are outliers
        outlier_pred = (iscore > self.threshold).astype(int)

        # populate output dict
        od = outlier_prediction_dict()
        od['meta'] = self.meta
        od['data']['is_outlier'] = outlier_pred
        if return_instance_score:
            od['data']['instance_score'] = iscore
        return od
