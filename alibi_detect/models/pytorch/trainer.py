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
        optimizer: torch.optim.optimizer = torch.optim.Adam,
        learning_rate: float = 1e-3,
        epochs: int = 20,
        verbose: bool = True,
) -> None:

    optimizer = optimizer(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        dl = tqdm(enumerate(dataloader), total=len(dataloader)) if verbose else enumerate(dataloader)
        for step, (x, y) in dl:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            optimizer.zero_grad()
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            if verbose:
                dl.set_description(f'Epoch {epoch + 1}/{epochs}')
                dl.set_postfix(dict(loss=loss.item()))
