import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Callable, Union


def trainer(
        model: Union[nn.Module, nn.Sequential],
        loss_fn: Callable,
        dataloader: DataLoader,
        device: torch.device,
        optimizer: Callable = torch.optim.Adam,
        learning_rate: float = 1e-3,
        preprocess_fn: Callable = None,
        epochs: int = 20,
        verbose: int = 1,
) -> None:
    """
    Train PyTorch model.

    Parameters
    ----------
    model
        Model to train.
    loss_fn
        Loss function used for training.
    dataloader
        PyTorch dataloader.
    optimizer
        Optimizer used for training.
    learning_rate
        Optimizer's learning rate.
    preprocess_fn
        Preprocessing function applied to each training batch.
    epochs
        Number of training epochs.
    verbose
        Whether to print training progress.
    """
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        dl = tqdm(enumerate(dataloader), total=len(dataloader)) if verbose == 1 else enumerate(dataloader)
        for step, (x, y) in dl:
            if isinstance(preprocess_fn, Callable):  # type: ignore
                x = preprocess_fn(x)
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            optimizer.zero_grad()  # type: ignore
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()  # type: ignore
            if verbose == 1:
                dl.set_description(f'Epoch {epoch + 1}/{epochs}')
                dl.set_postfix(dict(loss=loss.item()))
