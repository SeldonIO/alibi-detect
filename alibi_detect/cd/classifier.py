import logging
import numpy as np
from typing import Callable, Dict, Optional, Union
from alibi_detect.utils.frameworks import has_pytorch, has_tensorflow
from alibi_detect.utils.metrics import accuracy

if has_pytorch:
    from alibi_detect.cd.classifier_pt import ClassifierDriftTorch

if has_tensorflow:
    from alibi_detect.cd.classifier_tf import ClassifierDriftTF

logger = logging.getLogger(__name__)


class ClassifierDrift:

    def __init__(
            self,
            x_ref: np.ndarray,
            model: Callable,
            threshold: float = .55,
            preprocess_x_ref: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            metric_fn: Callable = accuracy,
            metric_name: Optional[str] = None,
            train_size: Optional[float] = .75,
            n_folds: Optional[int] = None,
            seed: int = 0,
            optimizer: Optional = None,
            learning_rate: float = 1e-3,
            compile_kwargs: Optional[dict] = None,
            batch_size: int = 32,
            epochs: int = 3,
            verbose: int = 0,
            train_kwargs: Optional[dict] = None,
            device: Optional[str] = None,
            data_type: Optional[str] = None
    ) -> None:
        super().__init__()

        backend = 'tensorflow' if hasattr(model, 'predict') else 'pytorch'
        if backend == 'tensorflow' and not has_tensorflow or backend == 'pytorch' and not has_pytorch:
            raise ImportError(f'{backend} not installed. Cannot initialize and run the '
                              f'ClassifierDrift detector with {backend} backend.')

        kwargs = locals()
        args = [kwargs['x_ref'], kwargs['model']]
        pop_kwargs = ['self', 'x_ref', 'model', 'backend', '__class__']
        if kwargs['optimizer'] is None:
            pop_kwargs += ['optimizer']
        [kwargs.pop(k, None) for k in pop_kwargs]

        if backend == 'tensorflow' and has_tensorflow:
            kwargs.pop('device', None)
            self._detector = ClassifierDriftTF(*args, **kwargs)
        else:
            kwargs.pop('compile_kwargs', None)
            self._detector = ClassifierDriftTorch(*args, **kwargs)

    def predict(self, x: np.ndarray, return_metric: bool = True) \
            -> Dict[Dict[str, str], Dict[str, Union[int, float]]]:
        """
        Predict whether a batch of data has drifted from the reference data.

        Parameters
        ----------
        x
            Batch of instances.
        return_metric
            Whether to return the drift metric from the detector.

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the drift prediction and optionally the drift metric and threshold.
        """
        return self._detector.predict(x, return_metric)
