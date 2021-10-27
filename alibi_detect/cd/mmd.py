import logging
import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple
from alibi_detect.utils.frameworks import has_pytorch, has_tensorflow

if has_pytorch:
    from alibi_detect.cd.pytorch.mmd import MMDDriftTorch

if has_tensorflow:
    from alibi_detect.cd.tensorflow.mmd import MMDDriftTF

logger = logging.getLogger(__name__)


class MMDDrift:
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            backend: str = 'tensorflow',
            p_val: float = .05,
            preprocess_x_ref: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            kernel: Callable = None,
            sigma: Optional[np.ndarray] = None,
            configure_kernel_from_x_ref: bool = True,
            n_permutations: int = 100,
            device: Optional[str] = None,
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
        preprocess_x_ref
            Whether to already preprocess and store the reference data.
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
        device
            Device type used. The default None tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either 'cuda', 'gpu' or 'cpu'. Only relevant for 'pytorch' backend.
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()

        backend = backend.lower()
        if backend == 'tensorflow' and not has_tensorflow or backend == 'pytorch' and not has_pytorch:
            raise ImportError(f'{backend} not installed. Cannot initialize and run the '
                              f'MMDDrift detector with {backend} backend.')
        elif backend not in ['tensorflow', 'pytorch']:
            raise NotImplementedError(f'{backend} not implemented. Use tensorflow or pytorch instead.')

        kwargs = locals()
        args = [kwargs['x_ref']]
        pop_kwargs = ['self', 'x_ref', 'backend', '__class__']
        [kwargs.pop(k, None) for k in pop_kwargs]

        if kernel is None:
            if backend == 'tensorflow':
                from alibi_detect.utils.tensorflow.kernels import GaussianRBF
            else:
                from alibi_detect.utils.pytorch.kernels import GaussianRBF  # type: ignore
            kwargs.update({'kernel': GaussianRBF})

        if backend == 'tensorflow' and has_tensorflow:
            kwargs.pop('device', None)
            self._detector = MMDDriftTF(*args, **kwargs)  # type: ignore
        else:
            self._detector = MMDDriftTorch(*args, **kwargs)  # type: ignore
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
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the drift prediction and optionally the p-value, threshold and MMD metric.
        """
        return self._detector.predict(x, return_p_val, return_distance)

    def score(self, x: Union[np.ndarray, list]) -> Tuple[float, float, np.ndarray]:
        """
        Compute the p-value resulting from a permutation test using the maximum mean discrepancy
        as a distance measure between the reference data and the data to be tested.

        Parameters
        ----------
        x
            Batch of instances.

        Returns
        -------
        p-value obtained from the permutation test, the MMD^2 between the reference and test set
        and the MMD^2 values from the permutation test.
        """
        return self._detector.score(x)
