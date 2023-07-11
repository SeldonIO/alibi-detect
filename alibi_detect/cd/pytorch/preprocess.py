from typing import Callable, Dict, Optional, Type, Union

import numpy as np
import torch
import torch.nn as nn
from alibi_detect.utils.pytorch.prediction import (predict_batch,
                                                   predict_batch_transformer)
from alibi_detect.utils._types import TorchDeviceType


class _Encoder(nn.Module):
    def __init__(
            self,
            input_layer: Optional[nn.Module],
            mlp: Optional[nn.Module] = None,
            input_dim: Optional[int] = None,
            enc_dim: Optional[int] = None,
            step_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_layer = input_layer
        if isinstance(mlp, nn.Module):
            self.mlp = mlp
        elif isinstance(enc_dim, int) and isinstance(step_dim, int):
            self.mlp = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, enc_dim + 2 * step_dim),
                nn.ReLU(),
                nn.Linear(enc_dim + 2 * step_dim, enc_dim + step_dim),
                nn.ReLU(),
                nn.Linear(enc_dim + step_dim, enc_dim)
            )
        else:
            raise ValueError('Need to provide either `enc_dim` and `step_dim` or a '
                             'nn.Module `mlp`')

    def forward(self, x: Union[np.ndarray, torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if self.input_layer is not None:
            x = self.input_layer(x)
        return self.mlp(x)


class UAE(nn.Module):
    def __init__(
            self,
            encoder_net: Optional[nn.Module] = None,
            input_layer: Optional[nn.Module] = None,
            shape: Optional[tuple] = None,
            enc_dim: Optional[int] = None
    ) -> None:
        super().__init__()
        is_enc = isinstance(encoder_net, nn.Module)
        is_enc_dim = isinstance(enc_dim, int)
        if is_enc:
            self.encoder = encoder_net
        elif not is_enc and is_enc_dim:  # set default encoder
            input_dim = np.prod(shape)
            step_dim = int((input_dim - enc_dim) / 3)
            self.encoder = _Encoder(input_layer, input_dim=input_dim, enc_dim=enc_dim, step_dim=step_dim)
        elif not is_enc and not is_enc_dim:
            raise ValueError('Need to provide either `enc_dim` or a nn.Module'
                             ' `encoder_net`.')

    def forward(self, x: Union[np.ndarray, torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        return self.encoder(x)


class HiddenOutput(nn.Module):
    def __init__(
            self,
            model: Union[nn.Module, nn.Sequential],
            layer: int = -1,
            flatten: bool = False
    ) -> None:
        super().__init__()
        layers = list(model.children())[:layer]
        if flatten:
            layers += [nn.Flatten()]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def preprocess_drift(x: Union[np.ndarray, list], model: Union[nn.Module, nn.Sequential],
                     device: TorchDeviceType = None, preprocess_batch_fn: Callable = None,
                     tokenizer: Optional[Callable] = None, max_len: Optional[int] = None,
                     batch_size: int = int(1e10), dtype: Union[Type[np.generic], torch.dtype] = np.float32) \
        -> Union[np.ndarray, torch.Tensor, tuple]:
    """
    Prediction function used for preprocessing step of drift detector.

    Parameters
    ----------
    x
        Batch of instances.
    model
        Model used for preprocessing.
    device
        Device type used. The default tries to use the GPU and falls back on CPU if needed.
        Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of
        ``torch.device``.
    preprocess_batch_fn
        Optional batch preprocessing function. For example to convert a list of objects to a batch which can be
        processed by the PyTorch model.
    tokenizer
        Optional tokenizer for text drift.
    max_len
        Optional max token length for text drift.
    batch_size
        Batch size used during prediction.
    dtype
        Model output type, e.g. np.float32 or torch.float32.

    Returns
    -------
    Numpy array or torch tensor with predictions.
    """
    if tokenizer is None:
        return predict_batch(x, model, device=device, batch_size=batch_size,
                             preprocess_fn=preprocess_batch_fn, dtype=dtype)
    else:
        return predict_batch_transformer(x, model, tokenizer, max_len, device=device,
                                         batch_size=batch_size, dtype=dtype)
