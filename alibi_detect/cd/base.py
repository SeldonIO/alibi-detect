from abc import abstractmethod
from functools import partial
import logging
import numpy as np
from typing import Callable, Dict, Optional, Tuple, Union
from alibi_detect.base import BaseDetector, concept_drift_dict
from alibi_detect.cd.preprocess import preprocess_drift
from alibi_detect.cd.utils import update_reference
from alibi_detect.utils.statstest import fdr

logger = logging.getLogger(__name__)


class BaseUnivariateDrift(BaseDetector):

    def __init__(self,
                 p_val: float = .05,
                 X_ref: Union[np.ndarray, list] = None,
                 preprocess_X_ref: bool = True,
                 update_X_ref: Optional[Dict[str, int]] = None,
                 preprocess_fn: Optional[Callable] = None,
                 preprocess_kwargs: Optional[dict] = None,
                 correction: str = 'bonferroni',
                 n_features: Optional[int] = None,
                 n_infer: int = 2,
                 input_shape: Optional[tuple] = None,
                 data_type: Optional[str] = None
                 ) -> None:
        """
        Generic drift detector component which serves as a base class for methods using
        univariate tests with multivariate correction.

        Parameters
        ----------
        p_val
            p-value used for significance of the statistical test for each feature. If the FDR correction method
            is used, this corresponds to the acceptable q-value.
        X_ref
            Data used as reference distribution. Can be a list for text data which is then turned into an array
            after the preprocessing step.
        preprocess_X_ref
            Whether to already preprocess and store the reference data.
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
        n_features
            Number of features used in the statistical test. No need to pass it if no preprocessing takes place.
            In case of a preprocessing step, this can also be inferred automatically but could be more
            expensive to compute.
        n_infer
            Number of instances used to infer number of features from.
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()

        if p_val is None:
            logger.warning('No p-value set for the drift threshold. Need to set it to detect data drift.')

        if isinstance(preprocess_kwargs, dict):
            if not isinstance(preprocess_fn, Callable):  # type: ignore
                preprocess_fn = preprocess_drift
            self.preprocess_fn = partial(preprocess_fn, **preprocess_kwargs)
            keys = list(preprocess_kwargs.keys())
        else:
            self.preprocess_fn, keys = None, []

        # optionally already preprocess reference data
        self.preprocess_X_ref = preprocess_X_ref
        if preprocess_X_ref and isinstance(self.preprocess_fn, Callable):  # type: ignore
            self.X_ref = self.preprocess_fn(X_ref)
        else:
            self.X_ref = X_ref
        self.update_X_ref = update_X_ref
        self.n = X_ref.shape[0]  # type: ignore
        self.p_val = p_val
        self.correction = correction

        if isinstance(input_shape, tuple):
            self.input_shape = input_shape
        elif 'max_len' in keys:
            self.input_shape = (preprocess_kwargs['max_len'],)
        elif isinstance(X_ref, np.ndarray):
            self.input_shape = X_ref.shape[1:]

        # compute number of features for the univariate tests
        if isinstance(n_features, int):
            self.n_features = n_features
        elif not isinstance(preprocess_fn, Callable) or preprocess_X_ref:
            # infer features from preprocessed reference data
            self.n_features = self.X_ref.reshape(self.X_ref.shape[0], -1).shape[-1]
        else:  # infer number of features after applying preprocessing step
            X = self.preprocess_fn(X_ref[0:min(X_ref.shape[0], n_infer)])
            self.n_features = X.reshape(X.shape[0], -1).shape[-1]

        if correction not in ['bonferroni', 'fdr'] and self.n_features > 1:
            raise ValueError('Only `bonferroni` and `fdr` are acceptable for multivariate correction.')

        # set metadata
        self.meta['detector_type'] = 'offline'  # offline refers to fitting the CDF for K-S
        self.meta['data_type'] = data_type

    def preprocess(self, X: Union[np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray]:
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
        if isinstance(self.preprocess_fn, Callable):  # type: ignore
            X = self.preprocess_fn(X)
            X_ref = self.X_ref if self.preprocess_X_ref else self.preprocess_fn(self.X_ref)
            return X_ref, X
        else:
            return self.X_ref, X

    @abstractmethod
    def feature_score(self, X_ref: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def score(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the feature-wise drift score which is the p-value of the
        statistical test and the test statistic.

        Parameters
        ----------
        X
            Batch of instances.

        Returns
        -------
        Feature level p-values and test statistics.
        """
        X_ref, X = self.preprocess(X)
        score, dist = self.feature_score(X_ref, X)  # feature-wise univariate test
        return score, dist

    def predict(self, X: Union[np.ndarray, list], drift_type: str = 'batch',
                return_p_val: bool = True, return_distance: bool = True) \
            -> Dict[Dict[str, str], Dict[str, Union[np.ndarray, int, float]]]:
        """
        Predict whether a batch of data has drifted from the reference data.

        Parameters
        ----------
        X
            Batch of instances.
        drift_type
            Predict drift at the 'feature' or 'batch' level. For 'batch', the test statistics for
            each feature are aggregated using the Bonferroni or False Discovery Rate correction.
        return_p_val
            Whether to return feature level p-values.
        return_distance
            Whether to return the test statistic between the features of the new batch and reference data.

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the drift prediction and optionally the feature level p-values,
         threshold after multivariate correction if needed and test statistics.
        """
        # compute drift scores
        p_vals, dist = self.score(X)

        # values below p-value threshold are drift
        if drift_type == 'feature':
            drift_pred = (p_vals < self.p_val).astype(int)
        elif drift_type == 'batch' and self.correction == 'bonferroni':
            threshold = self.p_val / self.n_features
            drift_pred = int((p_vals < threshold).any())
        elif drift_type == 'batch' and self.correction == 'fdr':
            drift_pred, threshold = fdr(p_vals, q_val=self.p_val)
        else:
            raise ValueError('`drift_type` needs to be either `feature` or `batch`.')

        # update reference dataset
        if (isinstance(self.update_X_ref, dict) and self.preprocess_fn is not None
                and self.preprocess_X_ref):
            X = self.preprocess_fn(X)
        self.X_ref = update_reference(self.X_ref, X, self.n, self.update_X_ref)
        # used for reservoir sampling
        self.n += X.shape[0]  # type: ignore

        # populate drift dict
        cd = concept_drift_dict()
        cd['meta'] = self.meta
        cd['data']['is_drift'] = drift_pred
        if return_p_val:
            cd['data']['p_val'] = p_vals
            cd['data']['threshold'] = self.p_val if drift_type == 'feature' else threshold
        if return_distance:
            cd['data']['distance'] = dist
        return cd
