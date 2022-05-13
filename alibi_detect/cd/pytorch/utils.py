import logging
from typing import Callable, Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)


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


def get_torch_device(device: Optional[str] = None) -> torch.device:
    if device is None or device.lower() in ['gpu', 'cuda']:
        torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch_device.type == 'cpu':
            logger.warning('No GPU detected, fall back on CPU.')
    else:
        torch_device = torch.device('cpu')

    return torch_device
