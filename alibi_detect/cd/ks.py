# for preprocess: pick from options or provide fn which outputs [batch, features] numpy array
import logging
import numpy as np
from scipy.stats import ks_2samp
from typing import Callable, Dict, Tuple
from alibi_detect.base import BaseDetector, concept_drift_dict
from alibi_detect.cd.utils import update_reference

logger = logging.getLogger(__name__)


class KSDrift(BaseDetector):

    def __init__(self,
                 threshold: float = None,
                 X_ref: np.ndarray = None,
                 update_X_ref: Dict[str, int] = None,
                 preprocess_fn: Callable = None,
                 preprocess_kwargs: dict = None,
                 alternative: str = 'two-sided',
                 data_type: str = None
                 ) -> None:
        """
        Kolmogorov-Smirnov (K-S) data drift detector with Bonferroni correction for multivariate data.

        Parameters
        ----------
        threshold
            p-value used for significance of the K-S test for each feature.
        X_ref
            Data used as reference distribution.
        update_X_ref
            Reference data can optionally be updated to the last n instances seen by the detector
            or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while
            for reservoir sampling {'reservoir_sampling': n} is passed.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
            Typically a dimensionality reduction technique.
        preprocess_kwargs
            Kwargs for `preprocess_fn`.
        alternative
            Defines the alternative hypothesis. Options are 'two-sided', 'less' or 'greater'.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()

        if threshold is None:
            logger.warning('No threshold level set. Need to set it to detect data drift.')

        self.threshold = threshold
        self.X_ref = X_ref
        self.update_X_ref = update_X_ref
        self.preprocess_fn = preprocess_fn
        self.preprocess_kwargs = preprocess_kwargs
        self.alternative = alternative
        self.n = X_ref.shape[0]

        # set metadata
        self.meta['detector_type'] = 'offline'  # offline refers to fitting the CDF for K-S
        self.meta['data_type'] = data_type

    def feature_score(self, X_ref: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Compute K-S scores per feature.

        Parameters
        ----------
        X_ref
            Reference instances to compare distribution with.
        X
            Batch of instances.

        Returns
        -------
        Feature level drift scores.
        """
        X = X.reshape(X.shape[0], -1)
        n_features = X.shape[1]
        p_val = np.zeros(n_features, dtype=np.float32)
        for f in range(n_features):
            p_val[f] = ks_2samp(X_ref[:, f], X[:, f], alternative=self.alternative)[1]
        return p_val

    def batch_score(self, fscore: np.ndarray) -> np.ndarray:
        # apply multivariate correction
        pass

    def score(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the feature-wise drift score which is the p-value of the Kolmogorov-Smirnov and
        then apply the Bonferroni correction across the features. Not that the p-value under H0 is
        uniformly distributed in [0,1]. As a result, no value should be attached to the size of p if
        H0 is not rejected.

        Parameters
        ----------
        X
            Batch of instances.

        Returns
        -------
        Feature and batch level drift scores.
        """
        # TODO: batch_size needed for preprocess_fn...? in kwargs!
        if isinstance(self.preprocess_fn, Callable):  # preprocess data
            X = self.preprocess_fn(X, **self.preprocess_kwargs)
            X_ref = self.preprocess_fn(self.X_ref, **self.preprocess_kwargs)
        else:
            X_ref = self.X_ref
        fscore = self.feature_score(X_ref, X)  # feature-wise K-S test
        bscore = self.batch_score(fscore)  # apply Bonferroni correction
        return fscore, bscore

    def predict(self,
                X: np.ndarray,
                drift_type: str = 'batch',
                return_feature_score: bool = True,
                return_batch_score: bool = True
                ) -> Dict[Dict[str, str], Dict[str, np.ndarray]]:
        """
        Predict whether a batch of data has drifted from the reference data.

        Parameters
        ----------
        X
            Batch of instances.
        drift_type
            Predict drift at the 'feature' or 'batch' level. For 'batch', the K-S statistics for
            each feature are aggregated using the Bonferroni correction.
        return_feature_score
            Whether to return feature level drift scores.
        return_batch_score
            Whether to return batch level drift scores

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the drift predictions and both feature and batch level drift scores.
        """
        # compute scores, check if above threshold, update reference data and return dict
        fscore, iscore = self.score(X)
        if drift_type == 'feature':
            drift_score = fscore
        elif drift_type == 'batch':
            drift_score = iscore
        else:
            raise ValueError('`drift_score` needs to be either `feature` or `batch`.')

        # values above threshold are drift
        drift_pred = (drift_score > self.threshold).astype(int)

        # update reference dataset
        self.X_ref = update_reference(self.X_ref, X, self.n, self.update_X_ref)
        self.n += X.shape[0]

        # populate drift dict
        cd = concept_drift_dict()
        cd['meta'] = self.meta
        cd['data']['is_drift'] = drift_pred
        if return_feature_score:
            cd['data']['feature_score'] = fscore
        if return_batch_score:
            cd['data']['batch_score'] = iscore
        return cd
