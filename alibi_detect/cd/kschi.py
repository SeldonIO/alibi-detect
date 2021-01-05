import logging
import numpy as np
from ks import KSDrift
from scipy.stats import ks_2samp, chisquare
from typing import Callable, Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class KSChiDrift(KSDrift):
    def __init__(
        self,
        p_val: float = 0.05,
        X_ref: Union[np.ndarray, list] = None,
        preprocess_X_ref: bool = True,
        update_X_ref: Optional[Dict[str, int]] = None,
        preprocess_fn: Optional[Callable] = None,
        preprocess_kwargs: Optional[dict] = None,
        correction: str = "bonferroni",
        alternative: str = "two-sided",
        n_features: Optional[int] = None,
        n_infer: int = 2,
        input_shape: Optional[tuple] = None,
        data_type: Optional[str] = None,
    ) -> None:
        """
        Adapted version of the base Kolmogorov-Smirnov (K-S) data drift detector with Bonferroni or
        False Discovery Rate (FDR) correction for multivariate data that allows a mix of continuous
        and discrete variables. Continous data continues to use the K-S significance test, while
        discrete variables use a Chi-square test.

        Parameters
        ----------
        p_val
            p-value used for significance of the K-S test for each feature. If the FDR correction method
            is used, this corresponds to the acceptable q-value.
        X_ref
            Data used as reference distribution.
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
        alternative
            Defines the alternative hypothesis. Options are 'two-sided', 'less' or 'greater'.
        n_features
            Number of features used in the K-S test. No need to pass it if no preprocessing takes place.
            In case of a preprocessing step, this can also be inferred automatically but could be more
            expensive to compute.
        n_infer
            Number of instances used to infer number of features from.
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__(
            p_val,
            X_ref,
            preprocess_X_ref,
            update_X_ref,
            preprocess_fn,
            preprocess_kwargs,
            correction,
            alternative,
            n_features,
            n_infer,
            input_shape,
            data_type,
        )

    def detect_var_type(self, thresh=0.05):
        """
        Predict whether a batch of data has drifted from the reference data.

        Parameters
        ----------
        thresh
            Percentage unique count of items must be less than to be considered as discrete
            variable.

        Returns
        -------
        List that matches size of x_ref.shape[1], where each item is either 'con', for
        continuous or 'dis' for discrete.
        """
        var_types = [
            "dis" if (np.unique(self.X_ref[:, n]).shape[0] / self.n) < 0.05 else "con"
            for n in range(self.n_features)
        ]

        return var_types

    def feature_score(
        self, X_ref: np.ndarray, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute K-S scores/Chi-Square and statistics per feature.

        Parameters
        ----------
        X_ref
            Reference instances to compare distribution with.
        X
            Batch of instances.

        Returns
        -------
        Feature level p-values and K-S/Chi-square statistics.
        """
        X = X.reshape(X.shape[0], -1)
        X_ref = X_ref.reshape(X_ref.shape[0], -1)
        p_val = np.zeros(self.n_features, dtype=np.float32)
        dist = np.zeros_like(p_val)

        var_types = self.detect_var_type()
        for f in range(self.n_features):
            if var_types[f] == "con":
                # TODO: update to 'exact' when bug fix is released in scipy 1.5
                dist[f], p_val[f] = ks_2samp(
                    X_ref[:, f], X[:, f], alternative=self.alternative, mode="asymp"
                )
            else:  ### chi square ###
                all_vals = np.unique(np.concatenate((a, b)))
                X_ref_cat = [(X_ref[:, f] == lab).sum() for lab in all_vals]
                X_cat = [(X[:, f] == lab).sum() for lab in all_vals]

                dist[f], p_val[f] = chisquare(X_cat, f_exp=X_ref_cat)

        return p_val, dist
