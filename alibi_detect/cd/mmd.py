import logging
import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple
from alibi_detect.utils.frameworks import has_pytorch, has_tensorflow, has_keops, BackendValidator, Framework
from alibi_detect.utils.warnings import deprecated_alias
from alibi_detect.base import DriftConfigMixin
from alibi_detect.utils._types import TorchDeviceType

if has_pytorch:
    from alibi_detect.cd.pytorch.mmd import MMDDriftTorch

if has_tensorflow:
    from alibi_detect.cd.tensorflow.mmd import MMDDriftTF

if has_keops and has_pytorch:
    from alibi_detect.cd.keops.mmd import MMDDriftKeops

logger = logging.getLogger(__name__)


class MMDDrift(DriftConfigMixin):
    @deprecated_alias(preprocess_x_ref='preprocess_at_init')
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            backend: str = 'tensorflow',
            p_val: float = .05,
            x_ref_preprocessed: bool = False,
            preprocess_at_init: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            kernel: Callable = None,
            sigma: Optional[np.ndarray] = None,
            configure_kernel_from_x_ref: bool = True,
            n_permutations: int = 100,
            batch_size_permutations: int = 1000000,
            device: TorchDeviceType = None,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Maximum Mean Discrepancy (MMD) data drift detector using a permutation test.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
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
        update_x_ref
            Reference data can optionally be updated to the last n instances seen by the detector
            or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while
            for reservoir sampling {'reservoir_sampling': n} is passed.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        kernel
            Kernel used for the MMD computation, defaults to Gaussian RBF kernel.
        sigma
            Optionally set the GaussianRBF kernel bandwidth. Can also pass multiple bandwidth values as an array.
            The kernel evaluation is then averaged over those bandwidths.
        configure_kernel_from_x_ref
            Whether to already configure the kernel bandwidth from the reference data.
        n_permutations
            Number of permutations used in the permutation test.
        batch_size_permutations
            KeOps computes the n_permutations of the MMD^2 statistics in chunks of batch_size_permutations.
            Only relevant for 'keops' backend.
        device
            Device type used. The default tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of
            ``torch.device``. Only relevant for 'pytorch' backend.
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()

        # Set config
        self._set_config(locals())

        backend = backend.lower()
        BackendValidator(
            backend_options={Framework.TENSORFLOW: [Framework.TENSORFLOW],
                             Framework.PYTORCH: [Framework.PYTORCH],
                             Framework.KEOPS: [Framework.KEOPS]},
            construct_name=self.__class__.__name__
        ).verify_backend(backend)

        kwargs = locals()
        args = [kwargs['x_ref']]
        pop_kwargs = ['self', 'x_ref', 'backend', '__class__']
        if backend == Framework.TENSORFLOW:
            pop_kwargs += ['device', 'batch_size_permutations']
            detector = MMDDriftTF
        elif backend == Framework.PYTORCH:
            pop_kwargs += ['batch_size_permutations']
            detector = MMDDriftTorch
        else:
            detector = MMDDriftKeops
        [kwargs.pop(k, None) for k in pop_kwargs]

        if kernel is None:
            if backend == Framework.TENSORFLOW:
                from alibi_detect.utils.tensorflow.kernels import GaussianRBF
            elif backend == Framework.PYTORCH:
                from alibi_detect.utils.pytorch.kernels import GaussianRBF  # type: ignore
            else:
                from alibi_detect.utils.keops.kernels import GaussianRBF  # type: ignore
            kwargs.update({'kernel': GaussianRBF})

        self._detector = detector(*args, **kwargs)
        self.meta = self._detector.meta

    def predict(self, x: Union[np.ndarray, list], return_p_val: bool = True, return_distance: bool = True) \
            -> Dict[Dict[str, str], Dict[str, Union[int, float]]]:
        """
        Predict whether a batch of data has drifted from the reference data.

        Parameters
        ----------
        x
            Batch of instances.
        return_p_val
            Whether to return the p-value of the permutation test.
        return_distance
            Whether to return the MMD metric between the new batch and reference data.

        Returns
        -------
        Dictionary containing ``'meta'`` and ``'data'`` dictionaries.
            - ``'meta'`` has the model's metadata.
            - ``'data'`` contains the drift prediction and optionally the p-value, threshold and MMD metric.
        """
        return self._detector.predict(x, return_p_val, return_distance)

    def score(self, x: Union[np.ndarray, list]) -> Tuple[float, float, float]:
        """
        Compute the p-value resulting from a permutation test using the maximum mean discrepancy
        as a distance measure between the reference data and the data to be tested.

        Parameters
        ----------
        x
            Batch of instances.

        Returns
        -------
        p-value obtained from the permutation test, the MMD^2 between the reference and test set, \
        and the MMD^2 threshold above which drift is flagged.
        """
        return self._detector.score(x)
