import numpy as np
from typing import Callable, Dict, Tuple, Optional, Union
from alibi_detect.cd.base import BaseUnivariateDrift
try:
    from scipy.stats import cramervonmises_2samp
except ImportError:
    cramervonmises_2samp = None


class CVMDrift(BaseUnivariateDrift):
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            p_val: float = .05,
            preprocess_x_ref: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            correction: str = 'bonferroni',
            n_features: Optional[int] = None,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Cramer-von Mises (CVM) data drift detector, which tests for any change in the distribution of continuous
        univariate data. For multivariate data, a separate CVM test is applied to each feature, and the obtained
        p-values are aggregated via the Bonferroni or False Discovery Rate (FDR) corrections.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        p_val
            p-value used for significance of the CVM test. If the FDR correction method
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
        n_features
            Number of features used in the CVM test. No need to pass it if no preprocessing takes place.
            In case of a preprocessing step, this can also be inferred automatically but could be more
            expensive to compute.
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        if cramervonmises_2samp is None:
            raise UserWarning("CVMDrift is only available if scipy version >= 1.7.0 installed.")
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

    def feature_score(self, x_ref: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs the two-sample Cramer-von Mises test(s), computing the p-value and test statistic per feature.

        Parameters
        ----------
        x_ref
            Reference instances to compare distribution with.
        x
            Batch of instances.

        Returns
        -------
        Feature level p-values and CVM statistics.
        """
        x = x.reshape(x.shape[0], -1)
        x_ref = x_ref.reshape(x_ref.shape[0], -1)
        p_val = np.zeros(self.n_features, dtype=np.float32)
        dist = np.zeros_like(p_val)
        for f in range(self.n_features):
            result = cramervonmises_2samp(x_ref[:, f], x[:, f], method='auto')
            p_val[f], dist[f] = result.pvalue, result.statistic
        return p_val, dist
