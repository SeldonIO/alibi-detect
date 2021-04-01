import logging
import numpy as np
from typing import Callable, Dict, Optional, Union
from alibi_detect.utils.frameworks import has_pytorch, has_tensorflow

if has_pytorch:
    from alibi_detect.cd.pytorch.mmd import MMDDriftTorch

if has_tensorflow:
    from alibi_detect.cd.tensorflow.mmd import MMDDriftTF

logger = logging.getLogger(__name__)


class MMDDrift:

    def __init__(
            self,
            x_ref: np.ndarray,
            backend: str = 'tensorflow',
            p_val: float = .05,
            preprocess_x_ref: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            kernel: Callable = None,
            sigma: Optional[np.ndarray] = None,
            infer_sigma: bool = True,
            n_permutations: int = 100,
            device: Optional[str] = None,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
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
                from alibi_detect.utils.pytorch.kernels import GaussianRBF
            kwargs.update({'kernel': GaussianRBF})

        if backend == 'tensorflow' and has_tensorflow:
            kwargs.pop('device', None)
            self._detector = MMDDriftTF(*args, **kwargs)
        else:
            self._detector = MMDDriftTorch(*args, **kwargs)

    def predict(self, x: np.ndarray, return_p_val: bool = True, return_distance: bool = True) \
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
