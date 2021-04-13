import logging
import numpy as np
from scipy.stats import chi2_contingency, ks_2samp
from typing import Callable, Dict, Optional, Tuple
from alibi_detect.cd.base import BaseUnivariateDrift

logger = logging.getLogger(__name__)


class TabularDrift(BaseUnivariateDrift):
    def __init__(
            self,
            x_ref: np.ndarray,
            p_val: float = .05,
            categories_per_feature: Dict[int, Optional[int]] = None,
            preprocess_x_ref: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            correction: str = 'bonferroni',
            alternative: str = 'two-sided',
            n_features: Optional[int] = None,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Mixed-type tabular data drift detector with Bonferroni or False Discovery Rate (FDR)
        correction for multivariate data. Kolmogorov-Smirnov (K-S) univariate tests are applied to
        continuous numerical data and Chi-Squared (Chi2) univariate tests to categorical data.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        p_val
            p-value used for significance of the K-S and Chi2 test for each feature.
            If the FDR correction method is used, this corresponds to the acceptable q-value.
        categories_per_feature
            Dict with as keys the column indices of the categorical features and as optional values
            the number of possible categories `n` for that feature. If left to None, all features are assumed
            to be continuous numerical. The column indices are post a potential preprocessing step.
            Eg: {0: 5, 1: 9, 2: 7} or {0: None, 1: None, 2: None}. In the latter case, the number of categories is
            inferred from the data. Categories are assumed to take values in the range `[0, 1, ..., n]`.
        preprocess_x_ref
            Whether to already preprocess and infer categories and frequencies for categorical reference data.
        update_x_ref
            Reference data can optionally be updated to the last n instances seen by the detector
            or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while
            for reservoir sampling {'reservoir_sampling': n} is passed.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
            Typically a dimensionality reduction technique.
        correction
            Correction type for multivariate data. Either 'bonferroni' or 'fdr' (False Discovery Rate).
        alternative
            Defines the alternative hypothesis for the K-S tests. Options are 'two-sided', 'less' or 'greater'.
        n_features
            Number of features used in the combined K-S/Chi-Squared tests. No need to pass it if
            no preprocessing takes place. In case of a preprocessing step, this can also be inferred
            automatically but could be more expensive to compute.
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__(
            x_ref=x_ref,
            p_val=p_val,
            preprocess_x_ref=preprocess_x_ref,
            update_x_ref=update_x_ref,
            preprocess_fn=preprocess_fn,
            correction=correction,
            n_features=n_features,
            input_shape=input_shape,
            data_type=data_type
        )
        self.alternative = alternative
        if isinstance(categories_per_feature, dict):
            # infer number of possible categories for each categorical feature from reference data
            if None in list(categories_per_feature.values()):
                x_flat = self.x_ref.reshape(self.x_ref.shape[0], -1)
                categories_per_feature = {f: x_flat[:, f].max().astype(int) + 1
                                          for f in categories_per_feature.keys()}
            self.categories_per_feature = categories_per_feature

            if update_x_ref is None and preprocess_x_ref:
                # already infer categories and frequencies for reference data
                self.x_ref_count = self._get_counts(x_ref)
            else:
                self.x_ref_count = None
        else:  # no categorical features assumed present
            self.categories_per_feature, self.x_ref_count = {}, None

    def feature_score(self, x_ref: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute K-S or Chi-Squared test statistics and p-values per feature.

        Parameters
        ----------
        x_ref
            Reference instances to compare distribution with.
        x
            Batch of instances.

        Returns
        -------
        Feature level p-values and K-S or Chi-Squared statistics.
        """
        x = x.reshape(x.shape[0], -1)
        x_ref = x_ref.reshape(x_ref.shape[0], -1)
        if self.categories_per_feature:
            x_count = self._get_counts(x)
            if not self.x_ref_count:  # compute categorical frequency counts for each feature
                x_ref_count = self._get_counts(x_ref)
            else:
                x_ref_count = self.x_ref_count
        p_val = np.zeros(self.n_features, dtype=np.float32)
        dist = np.zeros_like(p_val)
        for f in range(self.n_features):
            if f in list(self.categories_per_feature.keys()):
                contingency_table = np.vstack((x_ref_count[f], x_count[f]))
                dist[f], p_val[f], _, _ = chi2_contingency(contingency_table)
            else:
                dist[f], p_val[f] = ks_2samp(x_ref[:, f], x[:, f], alternative=self.alternative, mode='asymp')
        return p_val, dist

    def _get_counts(self, x: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Utility method for getting the counts of categories for each categorical variable.
        """
        return {f: np.bincount(x[:, f].astype(int), minlength=n_cat) for f, n_cat in
                self.categories_per_feature.items()}
