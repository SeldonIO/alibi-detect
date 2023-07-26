import logging
import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple
from alibi_detect.utils.frameworks import has_pytorch, has_tensorflow, BackendValidator, Framework
from alibi_detect.utils.warnings import deprecated_alias
from alibi_detect.base import DriftConfigMixin
from alibi_detect.utils._types import TorchDeviceType

if has_pytorch:
    from alibi_detect.cd.pytorch.context_aware import ContextMMDDriftTorch

if has_tensorflow:
    from alibi_detect.cd.tensorflow.context_aware import ContextMMDDriftTF

logger = logging.getLogger(__name__)


class ContextMMDDrift(DriftConfigMixin):
    @deprecated_alias(preprocess_x_ref='preprocess_at_init')
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            c_ref: np.ndarray,
            backend: str = 'tensorflow',
            p_val: float = .05,
            x_ref_preprocessed: bool = False,
            preprocess_at_init: bool = True,
            update_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            x_kernel: Callable = None,
            c_kernel: Callable = None,
            n_permutations: int = 1000,
            prop_c_held: float = 0.25,
            n_folds: int = 5,
            batch_size: Optional[int] = 256,
            device: TorchDeviceType = None,
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
        x_ref_preprocessed
            Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only
            the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference
            data will also be preprocessed.
        preprocess_at_init
            Whether to preprocess the reference data when the detector is instantiated. Otherwise, the reference
            data will be preprocessed at prediction time. Only applies if `x_ref_preprocessed=False`.
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
            Device type used. The default tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of
            ``torch.device``. Only relevant for 'pytorch' backend.
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        verbose
            Whether to print progress messages.
        """
        super().__init__()

        # Set config
        self._set_config(locals())

        backend = backend.lower()
        BackendValidator(
            backend_options={Framework.TENSORFLOW: [Framework.TENSORFLOW],
                             Framework.PYTORCH: [Framework.PYTORCH]},
            construct_name=self.__class__.__name__
        ).verify_backend(backend)

        kwargs = locals()
        args = [kwargs['x_ref'], kwargs['c_ref']]
        pop_kwargs = ['self', 'x_ref', 'c_ref', 'backend', '__class__']
        [kwargs.pop(k, None) for k in pop_kwargs]

        if x_kernel is None or c_kernel is None:
            if backend == Framework.TENSORFLOW:
                from alibi_detect.utils.tensorflow.kernels import GaussianRBF
            else:
                from alibi_detect.utils.pytorch.kernels import GaussianRBF  # type: ignore[assignment]
            if x_kernel is None:
                kwargs.update({'x_kernel': GaussianRBF})
            if c_kernel is None:
                kwargs.update({'c_kernel': GaussianRBF})

        if backend == Framework.TENSORFLOW:
            kwargs.pop('device', None)
            self._detector = ContextMMDDriftTF(*args, **kwargs)
        else:
            self._detector = ContextMMDDriftTorch(*args, **kwargs)
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
        Dictionary containing ``'meta'`` and ``'data'`` dictionaries.
            - ``'meta'`` has the model's metadata.
            - ``'data'`` contains the drift prediction and optionally the p-value, threshold, conditional MMD test \
            statistic and coupling matrices.
        """
        return self._detector.predict(x, c, return_p_val, return_distance, return_coupling)

    def score(self, x: Union[np.ndarray, list], c: np.ndarray) -> Tuple[float, float, float, Tuple]:
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
        p-value obtained from the conditional permutation test, the conditional MMD test statistic, the test \
        statistic threshold above which drift is flagged, and a tuple containing the coupling matrices \
        :math:`(W_{ref,ref}, W_{test,test}, W_{ref,test})`.
        """
        return self._detector.score(x, c)
