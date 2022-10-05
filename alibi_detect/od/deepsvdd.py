from typing import Callable, Optional

from functools import partial
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from alibi_detect.saving.registry import registry

from alibi_detect.od.backends import DeepSVDDTorch
from alibi_detect.utils.pytorch.data import TorchDataset
from alibi_detect.utils.pytorch.prediction import predict_batch
from alibi_detect.od.config import ConfigMixin, ModelWrapper
from alibi_detect.od.base import OutlierDetector
from alibi_detect.utils.frameworks import BackendValidator


backends = {
    'pytorch': DeepSVDDTorch,
}


@registry.register('KNN')
class DeepSVDD(OutlierDetector, ConfigMixin):
    CONFIG_PARAMS = ('model', 'weight_decay', 'optimizer', 'learning_rate', 'batch_size', 
                    'preprocess_batch_fn', 'epochs', 'verbose', 'train_kwargs', 'device', 
                    'dataset', 'dataloader', 'backend', 'input_shape')
    LARGE_PARAMS = ()
    BASE_OBJ = True
    FROM_PATH = ()

    def __init__(
        self,
        model: nn.Module,  # Should have no bias terms. Very important.
        input_shape: tuple = (1, 28, 28),
        weight_decay: float = 1e-3,
        optimizer: Callable = torch.optim.Adam,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        preprocess_batch_fn: Optional[Callable] = None,
        epochs: int = 10,
        verbose: int = 0,
        train_kwargs: Optional[dict] = None,
        device: Optional[str] = None,
        dataset: Callable = TorchDataset,
        dataloader: Callable = DataLoader,
        backend: str = 'pytorch'
    ) -> None:
        """
        DeepSVDD trains the supplied model to map training data onto a single point,
        which we take to be a vector of ones of dimension equal to that of the model
        output. The model should have no bias terms and all other parameters are regularised
        towards zero. This makes it non-trivial to map onto the point as the model being is encouraged
        to map onto the origin. If the model learns to map onto the n-d point it must have been by learning
        n invariant patterns in the data. One can assume these invariants to be independent as repeating the
        same pattern wouldn't be an effective used of regularised weight. It is assumed that points that 
        aren't from the same distribution will not satisfy these invariant patterns and therefore the distance
        to the point is given as an outlier score.
        Parameters
        ----------
        model:
            Should map data onto R^d where d is the desired number of invariants.
        weight_decay:
            The strength of the l2 regularisation of model weights.
        optimizer:
            Used to learn the model parameters
        The rest should be obvious.
        """
        model = ModelWrapper(model, input_shape)
        self._set_config(locals())
        backend = backend.lower()
        BackendValidator(
            backend_options={'pytorch': ['pytorch'],
                             'keops': ['keops']},
            construct_name=self.__class__.__name__
        ).verify_backend(backend)

        # define kwargs for dataloader and trainer
        self.backend = backends[backend](
            device,
            weight_decay,
            dataset, dataloader,
            batch_size,
            predict_batch,
            preprocess_batch_fn,
            optimizer,
            epochs,
            learning_rate,
            verbose
        )

        self.original_model = model
        self.model = model.copy()

    def fit(self, X: np.ndarray) -> None:
        self.backend.fit(self.model, self.original_model)
    
    def score(self, X: np.ndarray) -> np.ndarray:
        return self.backend.score(X)

    def _optimizer_serializer(self, key, val, path):
        return f'@{val.__module__}'

    def _dataset_serializer(self, key, val, path):
        return f'@{val.__module__}'

    def _dataloader_serializer(self, key, val, path):
        return f'@{val.__module__}'
