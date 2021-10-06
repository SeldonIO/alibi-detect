import logging
import numpy as np
from scipy.stats import fisher_exact
from typing import Callable, Dict, List, Tuple, Optional, Union
from alibi_detect.cd.base import BaseUnivariateDrift

logger = logging.getLogger(__name__)


class FETDrift(BaseUnivariateDrift):
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            p_val: float = .05,
            categories: Optional[Union[int, List[int]]] = None,
            preprocess_x_ref: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            alternative: str = 'two-sided',
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Fisher exact test (FET) data drift detector.

        Parameters
        ----------
        x_ref
            Data used as reference distribution. Data must consist of either [True, False]'s, or [0, 1]'s.
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
        alternative
            Defines the alternative hypothesis. Options are 'two-sided', 'less' or 'greater'.
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
        self.alternative = alternative

        # Preprocess reference data
        self.x_ref = self.x_ref.squeeze()  # squeeze in case of (n,1) array
        if self.x_ref.ndim != 1:
            raise ValueError("The `x_ref` data must be 1D for the FETDrift detector.")

        # Check data is only [False, True] or [0, 1]
        values = set(np.unique(x_ref))
        if values != {True, False} and values != {0, 1}:
            print(values)
            raise ValueError("The `x_ref` data must consist of only [0,1]'s or [False,True]'s for the "
                             "FETDrift detector.")

    def feature_score(self, x_ref: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs a Fisher exact test, computing the p-value and prior odds ratio.

        Parameters
        ----------
        x_ref
            Reference instances to compare distribution with. Data must consist of either [True, False]'s, or [0, 1]'s.
        x
            Batch of instances. Data must consist of either [True, False]'s, or [0, 1]'s.

        Returns
        -------
        The p-value and prior odds ratio.
        """
        # Preprocess data
        x = x.squeeze()  # squeeze in case of (n,1) array
        if x.ndim != 1:
            raise ValueError("The `x` data must be 1D for the FETDrift detector.")

        # Check data is only [False, True] or [0, 1]
        values = set(np.unique(x))
        if values != {True, False} and values != {0, 1}:
            raise ValueError("The `x` data must consist of only [0,1]'s or [False,True]'s for the FETDrift detector.")

        # Construct contingency table
        a = np.sum(x_ref)
        b = len(x_ref) - a
        c = np.sum(x)
        d = len(x) - c
        contingency_table = np.array([[a, b], [c, d]])

        # Apply test
        odds_ratio, p_val = fisher_exact(contingency_table, alternative=self.alternative)
        return p_val, odds_ratio  # type: ignore
