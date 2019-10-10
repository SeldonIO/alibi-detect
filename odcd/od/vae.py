import logging
import numpy as np

from odcd.models.autoencoder import TabularVAE
from odcd.models.trainer import train

logger = logging.getLogger(__name__)

# TODO: CategoricalVAE, TSVAE, ConvVAE


class TabularVAE:

    def __init__(self,
                 threshold: float,
                 score_type: str = 'mre'
                 ) -> None:

        self.threshold = threshold
        self.score_type = score_type

    def fit(self) -> None:
        # define VAE model -> import from 'models.autoencoder'
        # train -> import from 'models.trainer'
        pass

    def score(self, X: np.ndarray) -> np.ndarray:
        # check if latent proba ('proba') and/or reconstruction error ('mre') is used in score
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        outlier_score = self.score(X)  # compute outlier scores
        outlier_pred = (outlier_score > self.threshold).astype(int)  # convert outlier scores into predictions
        return outlier_pred
