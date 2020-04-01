import logging
import numpy as np
from scipy.stats import ks_2samp
from typing import Callable, Dict, Tuple
from alibi_detect.base import BaseDetector, concept_drift_dict
from alibi_detect.cd.utils import update_reference
from alibi_detect.utils.statstest import fdr

logger = logging.getLogger(__name__)


class KSDrift(BaseDetector):

    def __init__(self,
                 p_val: float = .05,
                 X_ref: np.ndarray = None,
                 update_X_ref: Dict[str, int] = None,
                 preprocess_fn: Callable = None,
                 preprocess_kwargs: dict = None,
                 correction: str = 'bonferroni',
                 alternative: str = 'two-sided',
                 n_features: int = None,
                 n_infer: int = 2,
                 data_type: str = None
                 ) -> None:
        """
        Kolmogorov-Smirnov (K-S) data drift detector with Bonferroni or False Discovery Rate (FDR)
        correction for multivariate data.

        Parameters
        ----------
        p_val
            p-value used for significance of the K-S test for each feature. If the FDR correction method
            is used, this corresponds to the acceptable q-value.
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
        correction
            Correction type for multivariate data. Either 'bonferroni' or 'fdr' (False Discovery Rate).
        alternative
            Defines the alternative hypothesis. Options are 'two-sided', 'less' or 'greater'.
        n_features
            Number of features used in the K-S test. No need to pass it if no preprocessing takes place.
            In case of a preprocessing step, this can also be inferred automatically but could be more
            expensive to compute.
        n_infer
            Number of instances used to infer number of features from.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()

        if p_val is None:
            logger.warning('No p-value set for the drift threshold. Need to set it to detect data drift.')

        self.X_ref = X_ref  # TODO: update rule for X_ref at init?
        self.update_X_ref = update_X_ref
        self.preprocess_fn = preprocess_fn
        self.preprocess_kwargs = preprocess_kwargs
        self.alternative = alternative
        self.n = X_ref.shape[0]
        self.p_val = p_val
        self.correction = correction

        # compute number of features for the K-S test
        if isinstance(n_features, int):
            self.n_features = n_features
        elif not isinstance(preprocess_fn, Callable):
            self.n_features = X_ref.reshape(X_ref.shape[0], -1).shape[-1]
        else:  # infer number of features after preprocessing step
            X = self.preprocess_fn(X_ref[0:min(X_ref.shape[0], n_infer)], **self.preprocess_kwargs)
            self.n_features = X.reshape(X.shape[0], -1).shape[-1]

        if correction not in ['bonferroni', 'fdr'] and self.n_features > 1:
            raise ValueError('Only `bonferroni` and `fdr` are acceptable for multivariate correction.')

        # set metadata
        self.meta['detector_type'] = 'offline'  # offline refers to fitting the CDF for K-S
        self.meta['data_type'] = data_type

    def preprocess(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Data preprocessing before computing the drift scores.

        Parameters
        ----------
        X
            Batch of instances.

        Returns
        -------
        Preprocessed reference data and new instances.
        """
        # TODO: check if makes sense to store preprocessed X_ref in attribute, don't think so for now
        if isinstance(self.preprocess_fn, Callable):  # type: ignore
            X = self.preprocess_fn(X, **self.preprocess_kwargs)
            X_ref = self.preprocess_fn(self.X_ref, **self.preprocess_kwargs)
            return X_ref, X
        else:
            return self.X_ref, X

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
        X_ref = X_ref.reshape(X_ref.shape[0], -1)
        p_val = np.zeros(self.n_features, dtype=np.float32)
        for f in range(self.n_features):
            # TODO: update to 'exact' when bug fix is released in scipy 1.5
            p_val[f] = ks_2samp(X_ref[:, f], X[:, f], alternative=self.alternative, mode='asymp')[1]
        return p_val

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the feature-wise drift score which is the p-value of the Kolmogorov-Smirnov.

        Parameters
        ----------
        X
            Batch of instances.

        Returns
        -------
        Feature level drift scores.
        """
        X_ref, X = self.preprocess(X)
        score = self.feature_score(X_ref, X)  # feature-wise K-S test
        return score

    def predict(self,
                X: np.ndarray,
                drift_type: str = 'batch',
                return_p_val: bool = True
                ) -> Dict[Dict[str, str], Dict[str, np.ndarray]]:
        """
        Predict whether a batch of data has drifted from the reference data.

        Parameters
        ----------
        X
            Batch of instances.
        drift_type
            Predict drift at the 'feature' or 'batch' level. For 'batch', the K-S statistics for
            each feature are aggregated using the Bonferroni or False Discovery Rate correction.
        return_p_val
            Whether to return feature level p-values.

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the drift predictions and both feature and batch level drift scores.
        """
        # compute drift scores
        p_vals = self.score(X)

        # values below p-value threshold are drift
        if drift_type == 'feature':  # undo multivariate correction
            drift_pred = (p_vals < self.p_val).astype(int)
        elif drift_type == 'batch' and self.correction == 'bonferroni':
            drift_pred = int((p_vals < self.p_val / self.n_features).any())
        elif drift_type == 'batch' and self.correction == 'fdr':
            drift_pred = int(fdr(p_vals, q_val=self.p_val))
        else:
            raise ValueError('`drift_type` needs to be either `feature` or `batch`.')

        # update reference dataset
        self.X_ref = update_reference(self.X_ref, X, self.n, self.update_X_ref)
        self.n += X.shape[0]  # used for reservoir sampling

        # populate drift dict
        cd = concept_drift_dict()
        cd['meta'] = self.meta
        cd['data']['is_drift'] = drift_pred
        if return_p_val:
            cd['data']['p_val'] = p_vals
        return cd
