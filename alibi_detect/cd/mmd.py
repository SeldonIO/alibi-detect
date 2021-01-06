import dask.array as da
from functools import partial
import logging
import numpy as np
from typing import Callable, Dict, Optional, Tuple, Union
from alibi_detect.base import BaseDetector, concept_drift_dict
from alibi_detect.cd.preprocess import preprocess_drift
from alibi_detect.cd.utils import update_reference
from alibi_detect.utils.distance import maximum_mean_discrepancy
from alibi_detect.utils.kernels import gaussian_kernel, infer_sigma
from alibi_detect.utils.statstest import permutation_test

logger = logging.getLogger(__name__)


class MMDDrift(BaseDetector):

    def __init__(self,
                 p_val: float = .05,
                 X_ref: Union[np.ndarray, list] = None,
                 preprocess_X_ref: bool = True,
                 update_X_ref: Optional[Dict[str, int]] = None,
                 preprocess_fn: Optional[Callable] = None,
                 preprocess_kwargs: Optional[dict] = None,
                 kernel: Callable = gaussian_kernel,
                 kernel_kwargs: Optional[dict] = None,
                 n_permutations: int = 1000,
                 chunk_size: Optional[int] = None,
                 input_shape: Optional[tuple] = None,
                 data_type: Optional[str] = None
                 ) -> None:
        """
        Maximum Mean Discrepancy (MMD) data drift detector using a permutation test.

        Parameters
        ----------
        p_val
            p-value used for the significance of the permutation test.
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
        kernel
            Kernel function used for the MMD.
        kernel_kwargs
            Kwargs for `kernel`
        n_permutations
            Number of permutations used in the permutation test.
        chunk_size
            Chunk size if dask is used to parallelise the computation.
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()

        if p_val is None:
            logger.warning('No p-value set for the drift threshold. Need to set it to detect data drift.')

        if isinstance(preprocess_kwargs, dict):  # type: ignore
            if not isinstance(preprocess_fn, Callable):  # type: ignore
                preprocess_fn = preprocess_drift
            self.preprocess_fn = partial(
                preprocess_fn,
                **preprocess_kwargs
            )
            keys = list(preprocess_kwargs.keys())
        else:
            self.preprocess_fn, keys = None, []

        # optionally already preprocess reference data
        self.preprocess_X_ref = preprocess_X_ref
        self.X_ref = self.preprocess_fn(X_ref) if preprocess_X_ref else X_ref
        self.update_X_ref = update_X_ref
        self.n = X_ref.shape[0]  # type: ignore
        self.p_val = p_val
        self.chunk_size = chunk_size

        if isinstance(input_shape, tuple):
            self.input_shape = input_shape
        elif 'max_len' in keys:
            self.input_shape = (preprocess_kwargs['max_len'],)
        elif isinstance(X_ref, np.ndarray):
            self.input_shape = X_ref.shape[1:]

        kwargs = kernel_kwargs if isinstance(kernel_kwargs, dict) else {}
        kwargs['kernel'] = kernel
        self.permutation_test = partial(
            permutation_test,
            n_permutations=n_permutations,
            metric=maximum_mean_discrepancy,
            return_distance=True,
            return_permutation_distance=True,
            **kwargs
        )

        permutation_args = list(self.permutation_test.keywords.keys())
        self.infer_sigma = (True if self.permutation_test.keywords['kernel'].__name__ == 'gaussian_kernel' and
                            'sigma' not in permutation_args else False)

        # set metadata
        self.meta['detector_type'] = 'offline'  # offline refers to fitting the CDF for K-S
        self.meta['data_type'] = data_type

    def preprocess(self, X: Union[np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray]:
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
            X = self.preprocess_fn(X)
            X_ref = self.X_ref if self.preprocess_X_ref else self.preprocess_fn(self.X_ref)
            return X_ref, X
        else:
            return self.X_ref, X

    def score(self, X: Union[np.ndarray, list]) -> Tuple[float, float, np.ndarray]:
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
        p_val, dist, dist_permutations = self.permutation_test(X_ref, X)  # type: ignore
        return p_val, dist, dist_permutations

    def predict(self, X: Union[np.ndarray, list], return_p_val: bool = True,
                return_distance: bool = True) -> Dict[Dict[str, str], Dict[str, Union[int, float]]]:
        """
        Predict whether a batch of data has drifted from the reference data.

        Parameters
        ----------
        X
            Batch of instances.
        return_p_val
            Whether to return the p-value of the permutation test.
        return_distance
            Whether to return the MMD metric between the new batch and reference data.

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the drift prediction and optionally the p-value, threshold and MMD metric.
        """
        # compute drift scores
        p_val, dist, dist_permutations = self.score(X)
        drift_pred = int(p_val < self.p_val)

        # compute distance threshold
        idx_threshold = int(self.p_val * len(dist_permutations))
        distance_threshold = np.sort(dist_permutations)[::-1][idx_threshold]

        # update reference dataset
        if (isinstance(self.update_X_ref, dict) and self.preprocess_fn is not None
                and self.preprocess_X_ref):
            X = self.preprocess_fn(X)
        self.X_ref = update_reference(self.X_ref, X, self.n, self.update_X_ref)
        # used for reservoir sampling
        self.n += X.shape[0]  # type: ignore

        # populate drift dict
        cd = concept_drift_dict()
        cd['meta'] = self.meta
        cd['data']['is_drift'] = drift_pred
        if return_p_val:
            cd['data']['p_val'] = p_val
            cd['data']['threshold'] = self.p_val
        if return_distance:
            cd['data']['distance'] = dist
            cd['data']['distance_threshold'] = distance_threshold
        return cd
