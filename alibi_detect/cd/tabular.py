import numpy as np
from scipy.stats import chi2_contingency, ks_2samp
from typing import Callable, Dict, List, Optional, Tuple, Union
from alibi_detect.cd.base import BaseUnivariateDrift


class TabularDrift(BaseUnivariateDrift):
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
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
            Dictionary with as keys the column indices of the categorical features and optionally as values
            the number of possible categorical values for that feature or a list with the possible values.
            If you know which features are categorical and simply want to infer the possible values of the
            categorical feature from the reference data you can pass a Dict[int, NoneType] such as
            {0: None, 3: None} if features 0 and 3 are categorical. If you also know how many categories are
            present for a given feature you could pass this in the `categories_per_feature` dict in the
            Dict[int, int] format, e.g. *{0: 3, 3: 2}*. If you pass N categories this will assume the possible
            values for the feature are [0, ..., N-1]. You can also explicitly pass the possible categories in the
            Dict[int, List[int]] format, e.g. {0: [0, 1, 2], 3: [0, 55]}. Note that the categories can be
            arbitrary int values.
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

        self.x_ref_categories, self.cat_vars = {}, []  # no categorical features assumed present
        if isinstance(categories_per_feature, dict):
            vals = list(categories_per_feature.values())
            int_types = (int, np.int16, np.int32, np.int64)
            if all(v is None for v in vals):  # categories_per_feature = Dict[int, NoneType]
                x_flat = self.x_ref.reshape(self.x_ref.shape[0], -1)
                categories_per_feature = {f: list(np.unique(x_flat[:, f]))  # type: ignore
                                          for f in categories_per_feature.keys()}
            elif all(isinstance(v, int_types) for v in vals):
                # categories_per_feature = Dict[int, int]
                categories_per_feature = {f: list(np.arange(v))  # type: ignore
                                          for f, v in categories_per_feature.items()}
            elif not all(isinstance(v, list) for v in vals) and \
                    all(isinstance(v, int_types) for val in vals for v in val):  # type: ignore
                raise ValueError('categories_per_feature needs to be None or one of '
                                 'Dict[int, NoneType], Dict[int, int], Dict[int, List[int]]')
            self.x_ref_categories = categories_per_feature
            self.cat_vars = list(self.x_ref_categories.keys())

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
        x_ref = x_ref.reshape(x_ref.shape[0], -1)
        x = x.reshape(x.shape[0], -1)

        # apply counts on union of categories per variable in both the reference and test data
        if self.cat_vars:
            x_categories = {f: list(np.unique(x[:, f])) for f in self.cat_vars}
            all_categories = {f: list(set().union(self.x_ref_categories[f], x_categories[f]))  # type: ignore
                              for f in self.cat_vars}
            x_ref_count = self._get_counts(x_ref, all_categories)
            x_count = self._get_counts(x, all_categories)

        p_val = np.zeros(self.n_features, dtype=np.float32)
        dist = np.zeros_like(p_val)
        for f in range(self.n_features):
            if f in self.cat_vars:
                contingency_table = np.vstack((x_ref_count[f], x_count[f]))
                dist[f], p_val[f], _, _ = chi2_contingency(contingency_table)
            else:
                dist[f], p_val[f] = ks_2samp(x_ref[:, f], x[:, f], alternative=self.alternative, mode='asymp')
        return p_val, dist

    def _get_counts(self, x: np.ndarray, categories: Dict[int, List[int]]) -> Dict[int, List[int]]:
        """
        Utility method for getting the counts of categories for each categorical variable.
        """
        return {f: [(x[:, f] == v).sum() for v in vals] for f, vals in categories.items()}
