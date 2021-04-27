import torch
from torch import nn
from typing import Callable


def activate_train_mode_for_dropout_layers(model: Callable) -> Callable:

    model.eval()  # type: ignore
    n_dropout_layers = 0
    for module in model.modules():  # type: ignore
        if isinstance(module, nn.Dropout):
            module.train()
            n_dropout_layers += 1
    if n_dropout_layers == 0:
        raise ValueError("No dropout layers identified.")

    return model


def zero_diag(mat: torch.tensor) -> torch.tensor:
    return mat - torch.diag(mat.diag())


def quantile(sample: torch.tensor, p: float, type: int = 7, sorted: bool = False) -> torch.tensor:
    """ See https://wikipedia.org/wiki/Quantile#Estimating_quantiles_from_a_sample """
    N = len(sample)
    if type == 6:  # With M = k*ert - 1 this one is exact
        h = (N+1)*p
    elif type == 7:
        h = (N-1)*p + 1
    elif type == 8:
        h = (N+1/3)*p + 1/3
    h_floor = int(h)
    if not sorted:
        sorted_sample = sample.sort().values
    quantile = sorted_sample[h_floor-1]
    if h_floor != h:
        quantile += (h - h_floor)*(sorted_sample[h_floor]-sorted_sample[h_floor-1])
    return quantile
