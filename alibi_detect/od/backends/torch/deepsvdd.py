import numpy as np
from functools import partial

import torch
from torch import nn
from copy import deepcopy
from typing import Callable, Optional

from alibi_detect.models.pytorch.trainer import trainer

class DeepSVDDTorch:

    def __init__(self, 
                 device, 
                 weight_decay, 
                 dataset, dataloader, 
                 batch_size, 
                 predict_batch, 
                 preprocess_batch_fn, 
                 optimizer,
                 epochs,
                 learning_rate, 
                 verbose):
        # set device, define model and training kwargs
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        # define kwargs for dataloader and trainer
        self.loss_fn = nn.MSELoss()
        self.reg_loss_fn = self._get_reg_loss_fn(weight_decay)
        self.dataset = dataset
        self.dataloader = partial(
            dataloader, 
            batch_size=batch_size, 
            shuffle=True)
        self.predict_fn = partial(
            predict_batch, 
            device=self.device,
            preprocess_fn=preprocess_batch_fn, 
            batch_size=batch_size
        )
        self.train_kwargs = {
            'optimizer': optimizer, 
            'epochs': epochs,  
            'preprocess_fn': preprocess_batch_fn,
            'learning_rate': learning_rate, 
            'verbose': verbose, 
            'reg_loss_fn': self.reg_loss_fn
        }

        if isinstance(self.train_kwargs, dict):
            self.train_kwargs.update(self.train_kwargs)


    def fit(self, 
            model, 
            original_model, 
            X: np.ndarray
        ) -> None:

        X = torch.as_tensor(X.astype(np.float32))
        demo_output = model(X[0:1])

        if demo_output.ndim != 2:
            raise ValueError("Model output should be 1d per instance.")

        sphere_dim = demo_output.shape[-1]
        y = torch.ones(len(X), sphere_dim)
        ds = self.dataset(X, y)
        dl = self.dataloader(ds)
        model = deepcopy(original_model)
        model = model.to(self.device)
        train_args = [model, self.loss_fn, dl, self.device]
        trainer(*train_args, **self.train_kwargs)  # type: ignore
        
    def score(self, X: np.ndarray) -> np.ndarray:
        X = torch.as_tensor(X.astype(np.float32))
        preds = torch.as_tensor(self.predict_fn(X, self.model.eval()))
        loss = nn.MSELoss(reduction='none')(preds, torch.ones_like(preds)).sum(-1)
        return loss.cpu().numpy()

    def _get_reg_loss_fn(self, weight_decay: float) -> Callable:
        def reg_loss_fn(model: nn.Module) -> torch.Tensor:
            loss = torch.tensor(0.0)
            for param in model.parameters():
                loss += param.square().sum()
            return weight_decay*loss
        return reg_loss_fn