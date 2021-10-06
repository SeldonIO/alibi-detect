import logging
import numpy as np
from scipy.stats import cramervonmises_2samp
from typing import Callable, Dict, Tuple, Optional, Union
from alibi_detect.cd.base import BaseUnivariateDrift

logger = logging.getLogger(__name__)


class CVMDrift(BaseUnivariateDrift):
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            p_val: float = .05,
            preprocess_x_ref: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Cramer-von Mises (CVM) data drift detector.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        p_val
            p-value used for the significance of the permutation test.
        preprocess_x_ref
            Whether to already preprocess and store the reference data.
        update_x_ref
            Reference data can optionally be updated to the last n instances seen by the detector
            or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while
            for reservoir sampling {'reservoir_sampling': n} is passed.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
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
            correction='None',
            n_features=1,
            input_shape=input_shape,
            data_type=data_type
        )

        # Preprocess reference data
        self.x_ref = self.x_ref.squeeze()  # squeeze in case of (n,1) array
        if self.x_ref.ndim != 1:
            raise ValueError("The `x_ref` data must be 1D for the CVMDrift detector.")

    def feature_score(self, x_ref: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs the two-sample Cramer-von Mises test, computing the p-value and test statistic.

        Parameters
        ----------
        x_ref
            Reference instances to compare distribution with.
        x
            Batch of instances.

        Returns
        -------
        The p-value and test statistic.
        """
        # Preprocess data
        x = x.squeeze()  # squeeze in case of (n,1) array
        if x.ndim != 1:
            raise ValueError("The `x` data must be 1D for the CVMDrift detector.")

        result = cramervonmises_2samp(x_ref, x, method='auto')
        return result.pvalue, result.statistic
