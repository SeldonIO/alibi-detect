import dask.array as da
from functools import partial
import logging
import numpy as np
from typing import Callable, Dict, Tuple
from alibi_detect.base import BaseDetector, concept_drift_dict
from alibi_detect.cd.utils import update_reference
from alibi_detect.utils.distance import maximum_mean_discrepancy
from alibi_detect.utils.kernels import gaussian_kernel, infer_sigma
from alibi_detect.utils.statstest import permutation_test

logger = logging.getLogger(__name__)


class MMDDrift(BaseDetector):

    def __init__(self,
                 p_val: float = .05,
                 X_ref: np.ndarray = None,
                 update_X_ref: Dict[str, int] = None,
                 preprocess_fn: Callable = None,
                 preprocess_kwargs: dict = None,
                 kernel: Callable = gaussian_kernel,
                 kernel_kwargs: dict = None,
                 n_permutations: int = 1000,
                 chunk_size: int = None,
                 data_type: str = None
                 ) -> None:
        """
        Maximum Mean Discrepancy (MMD) data drift detector using a permutation test.

        Parameters
        ----------
        p_val
            p-value used for the significance of the permutation test.
        X_ref
            Data used as reference distribution.
        update_X_ref
            Reference data can optionally be updated to the last n instances seen by the detector
            or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while
            for reservoir sampling {'reservoir_sampling': n} is passed.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
            Typically a dimensionality reduction technique.
        preprocess_kwargs
            Kwargs for `preprocess_fn`.
        kernel
            Kernel function used for the MMD.
        kernel_kwargs
            Kwargs for `kernel`
        n_permutations
            Number of permutations used in the permutation test.
        chunk_size
            Chunk size if dask is used to parallelise the computation.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()

        if p_val is None:
            logger.warning('No p-value set for the drift threshold. Need to set it to detect data drift.')

        self.X_ref = X_ref
        self.update_X_ref = update_X_ref
        self.preprocess_fn = preprocess_fn
        self.preprocess_kwargs = preprocess_kwargs
        self.n = X_ref.shape[0]
        self.p_val = p_val
        self.chunk_size = chunk_size

        kwargs = kernel_kwargs if isinstance(kernel_kwargs, dict) else {}
        kwargs['kernel'] = kernel
        self.permutation_test = partial(
            permutation_test,
            n_permutations=n_permutations,
            metric=maximum_mean_discrepancy,
            **kwargs
        )

        permutation_args = list(self.permutation_test.keywords.keys())
        self.infer_sigma = (True if self.permutation_test.keywords['kernel'].__name__ == 'gaussian_kernel' and
                            'sigma' not in permutation_args else False)

        # set metadata
        self.meta['detector_type'] = 'offline'  # offline refers to fitting the CDF for K-S
        self.meta['data_type'] = data_type

    def preprocess(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Data preprocessing before computing the drift scores.

        Parameters
        ----------
        X
            Batch of instances.

        Returns
        -------
        Preprocessed reference data and new instances.
        """
        if isinstance(self.preprocess_fn, Callable):  # type: ignore
            X = self.preprocess_fn(X, **self.preprocess_kwargs)
            X_ref = self.preprocess_fn(self.X_ref, **self.preprocess_kwargs)
            return X_ref, X
        else:
            return self.X_ref, X

    def score(self, X: np.ndarray) -> float:
        """
        Compute the p-value resulting from a permutation test using the maximum mean discrepancy
        as a distance measure between the reference data and the data to be tested.

        Parameters
        ----------
        X
            Batch of instances.

        Returns
        -------
        p-value obtained from the permutation test.
        """
        X_ref, X = self.preprocess(X)

        if isinstance(self.chunk_size, int):  # convert to dask arrays
            chunks = (self.chunk_size, X.shape[-1])
            X_ref = da.from_array(X_ref, chunks=chunks)
            X = da.from_array(X, chunks=chunks)

        if self.infer_sigma:
            sigma = infer_sigma(X_ref, X)
            self.permutation_test.keywords['sigma'] = np.array([sigma])
        p_val = self.permutation_test(X_ref, X)
        return p_val

    def predict(self, X: np.ndarray, return_p_val: bool = True) \
            -> Dict[Dict[str, str], Dict[str, int]]:
        """
        Predict whether a batch of data has drifted from the reference data.

        Parameters
        ----------
        X
            Batch of instances.
        return_p_val
            Whether to return the p-value of the permutation test.

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the drift prediction and optionally the p-value.
        """
        # compute drift scores
        p_val = self.score(X)
        drift_pred = int(p_val < self.p_val)

        # update reference dataset
        self.X_ref = update_reference(self.X_ref, X, self.n, self.update_X_ref)
        self.n += X.shape[0]  # used for reservoir sampling

        # populate drift dict
        cd = concept_drift_dict()
        cd['meta'] = self.meta
        cd['data']['is_drift'] = drift_pred
        if return_p_val:
            cd['data']['p_val'] = p_val
        return cd
