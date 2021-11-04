import logging
import numpy as np
from scipy.stats import hypergeom
from typing import Callable, Dict, Tuple, Optional, Union
from alibi_detect.cd.base import BaseUnivariateDrift

logger = logging.getLogger(__name__)


class FETDrift(BaseUnivariateDrift):
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            p_val: float = .05,
            preprocess_x_ref: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            correction: str = 'bonferroni',
            alternative: str = 'less',
            n_features: Optional[int] = None,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Fisher exact test (FET) data drift detector, with Bonferroni or False Discovery Rate (FDR)
        correction for multivariate data.
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
            Defines the alternative hypothesis. Options are 'less' or 'greater'.
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
        if alternative.lower() not in ['less', 'greater']:
            raise ValueError("`alternative` must be either 'less' or 'greater'.")
        self.alternative = alternative.lower()

        # Check data is only [False, True] or [0, 1]
        values = set(np.unique(x_ref))
        if values != {True, False} and values != {0, 1}:
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
        p_val = np.zeros(self.n_features, dtype=np.float32)
        odds_ratio = np.zeros_like(p_val)

        # Check data is only [False, True] or [0, 1]
        values = set(np.unique(x))
        if values != {True, False} and values != {0, 1}:
            raise ValueError("The `x` data must consist of only [0,1]'s or [False,True]'s for the FETDrift detector.")

        # Apply test per feature
        n_ref = x_ref.shape[0]
        n = x.shape[0]
        for f in range(self.n_features):
            sum_ref = np.sum(x_ref[:, f])
            sum_test = np.sum(x[:, f])
            if self.alternative == 'greater':
                p_val[f] = hypergeom.cdf(sum_ref, n_ref + n, sum_ref + sum_test, n_ref)
            else:
                p_val[f] = hypergeom.cdf(sum_test, n_ref + n, sum_ref + sum_test, n)

            odds_ratio[f] = (sum_test/(n-sum_test))/(sum_ref/(n_ref-sum_ref))

        return p_val, odds_ratio
