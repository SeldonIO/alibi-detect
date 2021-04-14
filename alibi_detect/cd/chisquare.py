import numpy as np
from scipy.stats import chi2_contingency
from typing import Callable, Dict, List, Optional, Tuple
from alibi_detect.cd.base import BaseUnivariateDrift


class ChiSquareDrift(BaseUnivariateDrift):
    def __init__(
            self,
            x_ref: np.ndarray,
            p_val: float = .05,
            categories_per_feature: Optional[Dict[int, int]] = None,
            preprocess_x_ref: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            correction: str = 'bonferroni',
            n_features: Optional[int] = None,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Chi-Squared data drift detector with Bonferroni or False Discovery Rate (FDR)
        correction for multivariate data.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        p_val
            p-value used for significance of the Chi-Squared test for each feature. If the FDR correction method
            is used, this corresponds to the acceptable q-value.
        categories_per_feature
            Dict with as keys the feature column index and as values the number of possible categorical
            values `n` for that feature. Eg: {0: 5, 1: 9, 2: 7}. If `None`, the number of categories is inferred
             from the data. Categories are assumed to take values in the range `[0, 1, ..., n]`.
        preprocess_x_ref
            Whether to already preprocess and infer categories and frequencies for reference data.
        update_x_ref
            Reference data can optionally be updated to the last n instances seen by the detector
            or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while
            for reservoir sampling {'reservoir_sampling': n} is passed.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
            Typically a dimensionality reduction technique.
        correction
            Correction type for multivariate data. Either 'bonferroni' or 'fdr' (False Discovery Rate).
        n_features
            Number of features used in the Chi-Squared test. No need to pass it if no preprocessing takes place.
            In case of a preprocessing step, this can also be inferred automatically but could be more
            expensive to compute.
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
        # construct categories from the user-specified dict
        if isinstance(categories_per_feature, dict):
            vals = list(categories_per_feature.values())
            if all(isinstance(v, (int, np.int16, np.int32, np.int64)) for v in vals):
                # categories_per_feature = Dict[int, int]
                categories_per_feature = {f: list(np.arange(v)) for f, v in categories_per_feature.items()}
            elif not all(isinstance(v, list) for v in vals):
                raise NotImplementedError  # categories_per_feature not Dict[int, list]
        else:  # infer number of possible categories for each feature from reference data
            x_flat = self.x_ref.reshape(self.x_ref.shape[0], -1)
            categories_per_feature = {f: list(np.unique(x_flat[:, f])) for f in range(self.n_features)}
        self.x_ref_categories = categories_per_feature

    def feature_score(self, x_ref: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Chi-Squared test statistic and p-values per feature.

        Parameters
        ----------
        x_ref
            Reference instances to compare distribution with.
        x
            Batch of instances.

        Returns
        -------
        Feature level p-values and Chi-Squared statistics.
        """
        x_ref = x_ref.reshape(x_ref.shape[0], -1)
        x = x.reshape(x.shape[0], -1)

        # apply counts on union of categories per variable in both the reference and test data
        x_categories = {f: list(np.unique(x[:, f])) for f in range(self.n_features)}
        all_categories = {f: list(set().union(self.x_ref_categories[f], x_categories[f]))
                          for f in range(self.n_features)}
        x_ref_count = self._get_counts(x_ref, all_categories)
        x_count = self._get_counts(x, all_categories)

        p_val = np.zeros(self.n_features, dtype=np.float32)
        dist = np.zeros_like(p_val)
        for f in range(self.n_features):  # apply Chi-Squared test
            contingency_table = np.vstack((x_ref_count[f], x_count[f]))
            dist[f], p_val[f], _, _ = chi2_contingency(contingency_table)
        return p_val, dist

    def _get_counts(self, x: np.ndarray, categories: Dict[int, List[int]]) -> Dict[int, List[int]]:
        """
        Utility method for getting the counts of categories for each categorical variable.
        """
        return {f: [(x[:, f] == v).sum() for v in vals] for f, vals in categories.items()}
