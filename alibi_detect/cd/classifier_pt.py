import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Callable, Dict, Optional, Union
from alibi_detect.cd.base import BaseClassifierDrift
from alibi_detect.models.pytorch.trainer import trainer
from alibi_detect.utils.metrics import accuracy
from alibi_detect.utils.pytorch.prediction import predict_batch


class ClassifierDriftTorch(BaseClassifierDrift):

    def __init__(
            self,
            x_ref: np.ndarray,
            model: Union[nn.Module, nn.Sequential],
            threshold: float = .55,
            preprocess_x_ref: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            metric_fn: Callable = accuracy,
            metric_name: Optional[str] = None,
            train_size: Optional[float] = .75,
            n_folds: Optional[int] = None,
            seed: int = 0,
            optimizer: torch.optim.optimizer = torch.optim.Adam,
            learning_rate: float = 1e-3,
            batch_size: int = 32,
            epochs: int = 3,
            verbose: int = 0,
            train_kwargs: Optional[dict] = None,
            device: Optional[str] = None,
            data_type: Optional[str] = None
    ) -> None:
        super().__init__(
            x_ref=x_ref,
            threshold=threshold,
            preprocess_x_ref=preprocess_x_ref,
            update_x_ref=update_x_ref,
            preprocess_fn=preprocess_fn,
            metric_fn=metric_fn,
            metric_name=metric_name,
            train_size=train_size,
            n_folds=n_folds,
            seed=seed,
            data_type=data_type
        )
        self.meta.update({'backend': 'pytorch'})

        # set device, define model and training kwargs
        if device is None or device.lower() in ['gpu', 'cuda']:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if self.device.type == 'cpu':
                print('No GPU detected, fall back on CPU.')
        else:
            self.device = torch.device('cpu')
        self.model = model

        # define kwargs for dataloader and trainer
        self.dl_kwargs = {'batch_size': batch_size, 'shuffle': True}
        self.train_kwargs = {'optimizer': optimizer, 'epochs': epochs,
                             'learning_rate': learning_rate, 'verbose': verbose}
        if isinstance(train_kwargs, dict):
            self.train_kwargs.update(train_kwargs)

    def score(self, x: np.ndarray) -> float:
        """
        Compute the out-of-fold drift metric such as the accuracy from a classifier
        trained to distinguish the reference data from the data to be tested.

        Parameters
        ----------
        x
            Batch of instances.

        Returns
        -------
        Drift metric (e.g. accuracy) obtained from out-of-fold predictions from a trained classifier.
        """
        x_ref, x = self.preprocess(x)
        x, y, splits = self.get_splits(x_ref, x)

        # iterate over folds: train a new model for each fold and make out-of-fold (oof) predictions
        preds_oof, idx_oof = [], []
        for idx_tr, idx_te in splits:
            x_tr, y_tr, x_te = x[idx_tr], np.eye(2)[y[idx_tr]], x[idx_te]
            ds_tr = TensorDataset(x_tr, y_tr)
            dl_tr = DataLoader(ds_tr, **self.dl_kwargs)
            # TODO: create copy of model but not of weights to pass to trainer + make sure predict does same
            train_args = [self.model, nn.CrossEntropyLoss(), dl_tr, self.device]
            trainer(*train_args, **self.train_kwargs)
            # TODO: use predict_batch fn for pytorch + change tf fit_kwargs
            preds = predict_batch(
                x_te, self.model, device=self.device, batch_size=self.dl_kwargs['batch_size']
            )
            preds_oof.append(preds)
            idx_oof.append(idx_te)
        preds_oof = np.concatenate(preds_oof, axis=0)[:, 1]
        idx_oof = np.concatenate(idx_oof, axis=0)
        drift_metric = self.metric_fn(y[idx_oof], preds_oof)
        return drift_metric
