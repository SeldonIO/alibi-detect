import logging
import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple, Type
from alibi_detect.utils.frameworks import has_pytorch, has_tensorflow
from alibi_detect.cd.domain_clf import DomainClf, SVCDomainClf

if has_pytorch:
    from alibi_detect.cd.pytorch.context_aware import ContextAwareDriftTorch

if has_tensorflow:
    from alibi_detect.cd.tensorflow.context_aware import ContextAwareDriftTF

logger = logging.getLogger(__name__)


class ContextAwareDrift:
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            c_ref: np.ndarray,
            backend: str = 'pytorch',  # TODO - change to tf once implemented
            p_val: float = .05,
            preprocess_x_ref: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            x_kernel: Callable = None,
            c_kernel: Callable = None,
            domain_clf: Type[DomainClf] = SVCDomainClf,
            n_permutations: int = 1000,
            cond_prop: float = 0.25,
            lams: Optional[Tuple[float, float]] = None,
            batch_size: Optional[int] = 256,
            device: Optional[str] = None,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None,
            verbose: bool = False
    ) -> None:
        """
        Maximum Mean Discrepancy (MMD) based context aware drift detector.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        c_ref
            Data used as context for the reference distribution.
        backend
            Backend used for the MMD implementation.
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
        x_kernel
            Kernel defined on the input data, defaults to Gaussian RBF kernel.
        c_kernel
            Kernel defined on the context data, defaults to Gaussian RBF kernel.
        domain_clf
            Domain classifier, takes conditioning variables and their domain, and returns propensity scores (probs of
            being test instances). Must be a subclass of DomainClf. # TODO - add link
        n_permutations
            Number of permutations used in the permutation test.
        cond_prop
            Proportion of contexts held out to condition on.
        lams
            Ref and test regularisation parameters. Tuned if None.
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
                              f'ContextAwareDrift detector with {backend} backend.')
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
            kwargs.update({'x_kernel': GaussianRBF()} if x_kernel is None else {'x_kernel': x_kernel})  # type: ignore
            kwargs.update({'c_kernel': GaussianRBF()} if c_kernel is None else {'c_kernel': c_kernel})  # type: ignore

        if backend == 'tensorflow' and has_tensorflow:
            kwargs.pop('device', None)
            self._detector = ContextAwareDriftTF(*args, **kwargs)  # type: ignore
        else:
            self._detector = ContextAwareDriftTorch(*args, **kwargs)  # type: ignore
        self.meta = self._detector.meta

    def predict(self, x: Union[np.ndarray, list], c: np.ndarray,
                return_p_val: bool = True, return_distance: bool = True, return_coupling: bool = False) \
            -> Dict[Dict[str, str], Dict[str, Union[int, float]]]:
        """
        Predict whether a batch of data has drifted from the reference data. # TODO

        Parameters
        ----------
        x
            Batch of instances.
        c
            Context associated with batch of instances.
        return_p_val
            Whether to return the p-value of the permutation test.
        return_distance
            Whether to return the MMD metric between the new batch and reference data.
        return_coupling
            Whether to return the coupling matrices.

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the drift prediction and optionally the p-value, threshold, MMD metric and coupling matrices.
        """
        return self._detector.predict(x, c, return_p_val, return_distance, return_coupling)

    def score(self, x: Union[np.ndarray, list], c: np.ndarray) -> Tuple[float, float, np.ndarray, Tuple]:
        """
        Compute the p-value resulting from a permutation test using the maximum mean discrepancy
        as a distance measure between the reference data and the data to be tested.  # TODO

        Parameters
        ----------
        x
            Batch of instances.
        c
            Context associated with batch of instances.

        Returns
        -------
        p-value obtained from the permutation test, the MMD^2 between the reference and test set, the
        MMD^2 values from the permutation test, and the coupling matrices.
        """
        return self._detector.score(x, c)
