import numpy as np
from typing import Callable, Dict, Tuple, Optional, Union
from alibi_detect.cd.base import BaseUnivariateDrift
from scipy.stats import fisher_exact


class FETDrift(BaseUnivariateDrift):
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            p_val: float = .05,
            preprocess_x_ref: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            correction: str = 'bonferroni',
            alternative: str = 'decrease',
            n_features: Optional[int] = None,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Fisher exact test (FET) data drift detector, which tests for a change in the mean of binary univariate data.
        For multivariate data, a separate FET test is applied to each feature, and the obtained p-values are
        aggregated via the Bonferroni or False Discovery Rate (FDR) corrections.

        Parameters
        ----------
        x_ref
            Data used as reference distribution. Data must consist of either [True, False]'s, or [0, 1]'s.
        p_val
            p-value used for significance of the FET test. If the FDR correction method
            is used, this corresponds to the acceptable q-value.
        preprocess_x_ref
            Whether to already preprocess and store the reference data.
        update_x_ref
            Reference data can optionally be updated to the last n instances seen by the detector
            or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while
            for reservoir sampling {'reservoir_sampling': n} is passed.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        correction
            Correction type for multivariate data. Either 'bonferroni' or 'fdr' (False Discovery Rate).
        alternative
            Defines the alternative hypothesis. Options are 'greater', 'less' or `two-sided`. These correspond to
            an increase, decrease, or any change in the mean of the Bernoulli data.
        n_features
            Number of features used in the FET test. No need to pass it if no preprocessing takes place.
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
        if alternative.lower() not in ['greater', 'less', 'two-sided']:
            raise ValueError("`alternative` must be either 'greater', 'less' or 'two-sided'.")
        self.alternative = alternative.lower()

        # Check data is only [False, True] or [0, 1]
        values = set(np.unique(x_ref))
        if not set(values).issubset(['0', '1', True, False]):
            raise ValueError("The `x_ref` data must consist of only (0,1)'s or (False,True)'s for the "
                             "FETDrift detector.")

    def feature_score(self, x_ref: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs Fisher exact test(s), computing the p-value per feature.

        Parameters
        ----------
        x_ref
            Reference instances to compare distribution with. Data must consist of either [True, False]'s, or [0, 1]'s.
        x
            Batch of instances. Data must consist of either [True, False]'s, or [0, 1]'s.

        Returns
        -------
        Feature level p-values and odds ratios.
        """
        x = x.reshape(x.shape[0], -1).astype(dtype=np.int64)
        x_ref = x_ref.reshape(x_ref.shape[0], -1).astype(dtype=np.int64)

        # Check data is only [False, True] or [0, 1]
        values = set(np.unique(x))
        if not set(values).issubset(['0', '1', True, False]):
            raise ValueError("The `x` data must consist of only [0,1]'s or [False,True]'s for the FETDrift detector.")

        # Perform FET for each feature
        n_ref = x_ref.shape[0]
        n = x.shape[0]
        sum_ref = np.sum(x_ref, axis=0)
        sum_test = np.sum(x, axis=0)
        p_val = np.empty(self.n_features)
        odds_ratio = np.empty_like(p_val)
        for f in range(self.n_features):
            table = np.array([[sum_test[f], sum_ref[f]], [n - sum_test[f], n_ref - sum_ref[f]]])
            odds_ratio[f], p_val[f] = fisher_exact(table, alternative=self.alternative)

        return p_val, odds_ratio
