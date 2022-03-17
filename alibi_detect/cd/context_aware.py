import logging
import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple
from alibi_detect.utils.frameworks import has_pytorch, has_tensorflow

if has_pytorch:
    from alibi_detect.cd.pytorch.context_aware import ContextMMDDriftTorch

if has_tensorflow:
    from alibi_detect.cd.tensorflow.context_aware import ContextMMDDriftTF

logger = logging.getLogger(__name__)


class ContextMMDDrift:
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            c_ref: np.ndarray,
            backend: str = 'tensorflow',
            p_val: float = .05,
            preprocess_x_ref: bool = True,
            update_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            x_kernel: Callable = None,
            c_kernel: Callable = None,
            n_permutations: int = 1000,
            prop_c_held: float = 0.25,
            n_folds: int = 5,
            batch_size: Optional[int] = 256,
            device: Optional[str] = None,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None,
            verbose: bool = False
    ) -> None:
        """
        A context-aware drift detector based on a conditional analogue of the maximum mean discrepancy (MMD).
        Only detects differences between samples that can not be attributed to differences between associated
        sets of contexts. p-values are computed using a conditional permutation test.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        c_ref
            Context for the reference distribution.
        backend
            Backend used for the MMD implementation.
        p_val
            p-value used for the significance of the permutation test.
        preprocess_x_ref
            Whether to already preprocess and store the reference data `x_ref`.
        update_ref
            Reference data can optionally be updated to the last N instances seen by the detector.
            The parameter should be passed as a dictionary *{'last': N}*.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        x_kernel
            Kernel defined on the input data, defaults to Gaussian RBF kernel.
        c_kernel
            Kernel defined on the context data, defaults to Gaussian RBF kernel.
        n_permutations
            Number of permutations used in the permutation test.
        prop_c_held
            Proportion of contexts held out to condition on.
        n_folds
            Number of cross-validation folds used when tuning the regularisation parameters.
        batch_size
            If not None, then compute batches of MMDs at a time (rather than all at once).
        device
            Device type used. The default None tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either 'cuda', 'gpu' or 'cpu'. Only relevant for 'pytorch' backend.
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        verbose
            Whether to print progress messages.
        """
        super().__init__()

        backend = backend.lower()
        if backend == 'tensorflow' and not has_tensorflow or backend == 'pytorch' and not has_pytorch:
            raise ImportError(f'{backend} not installed. Cannot initialize and run the '
                              f'ContextMMMDrift detector with {backend} backend.')
        elif backend not in ['tensorflow', 'pytorch']:
            raise NotImplementedError(f'{backend} not implemented. Use tensorflow or pytorch instead.')

        kwargs = locals()
        args = [kwargs['x_ref'], kwargs['c_ref']]
        pop_kwargs = ['self', 'x_ref', 'c_ref', 'backend', '__class__']
        [kwargs.pop(k, None) for k in pop_kwargs]

        if x_kernel is None or c_kernel is None:
            if backend == 'tensorflow':
                from alibi_detect.utils.tensorflow.kernels import GaussianRBF
            else:
                from alibi_detect.utils.pytorch.kernels import GaussianRBF  # type: ignore[no-redef]
            if x_kernel is None:
                kwargs.update({'x_kernel': GaussianRBF})
            if c_kernel is None:
                kwargs.update({'c_kernel': GaussianRBF})

        if backend == 'tensorflow' and has_tensorflow:
            kwargs.pop('device', None)
            self._detector = ContextMMDDriftTF(*args, **kwargs)  # type: ignore
        else:
            self._detector = ContextMMDDriftTorch(*args, **kwargs)  # type: ignore
        self.meta = self._detector.meta

    def predict(self, x: Union[np.ndarray, list], c: np.ndarray,
                return_p_val: bool = True, return_distance: bool = True, return_coupling: bool = False) \
            -> Dict[Dict[str, str], Dict[str, Union[int, float]]]:
        """
        Predict whether a batch of data has drifted from the reference data, given the provided context.

        Parameters
        ----------
        x
            Batch of instances.
        c
            Context associated with batch of instances.
        return_p_val
            Whether to return the p-value of the permutation test.
        return_distance
            Whether to return the conditional MMD test statistic between the new batch and reference data.
        return_coupling
            Whether to return the coupling matrices.

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the drift prediction and optionally the p-value, threshold, conditional MMD test statistic
        and coupling matrices.
        """
        return self._detector.predict(x, c, return_p_val, return_distance, return_coupling)

    def score(self, x: Union[np.ndarray, list], c: np.ndarray) -> Tuple[float, float, np.ndarray, Tuple]:
        """
        Compute the MMD based conditional test statistic, and perform a conditional permutation test to obtain a
        p-value representing the test statistic's extremity under the null hypothesis.

        Parameters
        ----------
        x
            Batch of instances.
        c
            Context associated with batch of instances.

        Returns
        -------
        p-value obtained from the conditional permutation test, the conditional MMD test statistic, the permuted
        test statistics, and a tuple containing the coupling matrices (Wref,ref, Wtest,test, Wref,test).
        """
        return self._detector.score(x, c)
