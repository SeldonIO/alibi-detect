import logging
import numpy as np
import tensorflow as tf
from typing import Callable, Dict, Tuple
from odcd.models.autoencoder import AEGMM, eucl_cosim_features
from odcd.models.trainer import trainer
from odcd.od.base import BaseOutlierDetector, FitMixin, ThresholdMixin, outlier_prediction_dict

logger = logging.getLogger(__name__)


class OutlierAEGMM(BaseOutlierDetector, FitMixin, ThresholdMixin):

    def __init__(self,
                 threshold: float = None,
                 aegmm: tf.keras.Model = None,
                 encoder_net: tf.keras.Sequential = None,
                 decoder_net: tf.keras.Sequential = None,
                 gmm_density_net: tf.keras.Sequential = None,
                 n_gmm: int = None,
                 recon_features: Callable = eucl_cosim_features,
                 data_type: str = None
                 ) -> None:
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
            raise TypeError('No valid format detected for `vae` (tf.keras.Model) '
                            'or `encoder_net`, `decoder_net` and `gmm_density_net` (tf.keras.Sequential).')

        # set metadata
        self.meta['detector_type'] = 'offline'
        self.meta['data_type'] = data_type

    def fit(self):
        pass

    def infer_threshold(self, X: np.ndarray) -> None:
        pass

    def score(self):
        pass

    def predict(self):
        pass
