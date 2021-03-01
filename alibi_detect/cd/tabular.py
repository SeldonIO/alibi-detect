import logging
import numpy as np
from scipy.stats import chisquare, ks_2samp
from typing import Callable, Dict, Optional, Tuple
from alibi_detect.cd.base import BaseUnivariateDrift

logger = logging.getLogger(__name__)


class TabularDrift(BaseUnivariateDrift):

    def __init__(self,
                 p_val: float = .05,
                 X_ref: Optional[np.ndarray] = None,
                 categories_per_feature: Dict[int, Optional[int]] = None,
                 preprocess_X_ref: bool = True,
                 update_X_ref: Optional[Dict[str, int]] = None,
                 preprocess_fn: Optional[Callable] = None,
                 preprocess_kwargs: Optional[dict] = None,
                 correction: str = 'bonferroni',
                 alternative: str = 'two-sided',
                 n_features: Optional[int] = None,
                 n_infer: int = 2,
                 input_shape: Optional[tuple] = None,
                 data_type: Optional[str] = None
                 ) -> None:
        """
        Mixed-type tabular data drift detector with Bonferroni or False Discovery Rate (FDR)
        correction for multivariate data. Kolmogorov-Smirnov (K-S) univariate tests are applied to
        continuous numerical data and Chi-Squared (Chi2) univariate tests to categorical data.

        Parameters
        ----------
        p_val
            p-value used for significance of the K-S and Chi2 test for each feature.
            If the FDR correction method is used, this corresponds to the acceptable q-value.
        X_ref
            Data used as reference distribution.
        categories_per_feature
            Dict with as keys the column indices of the categorical features and as optional values
            the number of possible categories `n` for that feature. If left to None, all features are assumed
            to be continuous numerical. The column indices are post a potential preprocessing step.
            Eg: {0: 5, 1: 9, 2: 7} or {0: None, 1: None, 2: None}. In the latter case, the number of categories is
            inferred from the data. Categories are assumed to take values in the range `[0, 1, ..., n]`.
        preprocess_X_ref
            Whether to already preprocess and infer categories and frequencies for categorical reference data.
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
            Defines the alternative hypothesis for the K-S tests. Options are 'two-sided', 'less' or 'greater'.
        n_features
            Number of features used in the combined K-S/Chi-Squared tests. No need to pass it if
            no preprocessing takes place. In case of a preprocessing step, this can also be inferred
            automatically but could be more expensive to compute.
        n_infer
            Number of instances used to infer number of features from.
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__(
            p_val=p_val,
            X_ref=X_ref,
            preprocess_X_ref=preprocess_X_ref,
            update_X_ref=update_X_ref,
            preprocess_fn=preprocess_fn,
            preprocess_kwargs=preprocess_kwargs,
            correction=correction,
            n_features=n_features,
            n_infer=n_infer,
            input_shape=input_shape,
            data_type=data_type
        )
        self.alternative = alternative
        if isinstance(categories_per_feature, dict):
            # infer number of possible categories for each categorical feature from reference data
            if None in list(categories_per_feature.values()):
                X_flat = self.X_ref.reshape(self.X_ref.shape[0], -1)
                categories_per_feature = {f: X_flat[:, f].max().astype(int) + 1 for f in range(self.n_features)}
            self.categories_per_feature = categories_per_feature

            if update_X_ref is None and preprocess_X_ref:
                # already infer categories and frequencies for reference data
                self.X_ref_count = self._get_counts(X_ref)
            else:
                self.X_ref_count = None
        else:  # no categorical features assumed present
            self.categories_per_feature, self.X_ref_count = {}, None

    def feature_score(self, X_ref: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute K-S or Chi-Squared test statistics and p-values per feature.

        Parameters
        ----------
        X_ref
            Reference instances to compare distribution with.
        X
            Batch of instances.

        Returns
        -------
        Feature level p-values and K-S or Chi-Squared statistics.
        """
        X = X.reshape(X.shape[0], -1)
        X_ref = X_ref.reshape(X_ref.shape[0], -1)
        if self.categories_per_feature:
            X_count = self._get_counts(X)

            if not self.X_ref_count:  # compute categorical frequency counts for each feature
                X_ref_count = self._get_counts(X_ref)
            else:
                X_ref_count = self.X_ref_count
        p_val = np.zeros(self.n_features, dtype=np.float32)
        dist = np.zeros_like(p_val)
        for f in range(self.n_features):
            if f in list(self.categories_per_feature.keys()):
                n_ref, n_obs = X_ref_count[f].sum(), X_count[f].sum()
                dist[f], p_val[f] = chisquare(X_count[f], f_exp=X_ref_count[f] * n_obs / n_ref)
            else:
                dist[f], p_val[f] = ks_2samp(X_ref[:, f], X[:, f], alternative=self.alternative, mode='asymp')
        return p_val, dist

    def _get_counts(self, X: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Utility method for getting the counts of categories for each categorical variable.
        """
        return {f: np.bincount(X[:, f].astype(int), minlength=n_cat) for f, n_cat in
                self.categories_per_feature.items()}
