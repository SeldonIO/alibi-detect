import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Optional, Union
from alibi_detect.utils.pytorch.prediction import predict_batch, predict_batch_transformer


class HiddenOutput(nn.Module):
    def __init__(
            self,
            model: Union[nn.Module, nn.Sequential],
            layer: int = -1
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(*list(model.children())[:layer])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def preprocess_drift(x: np.ndarray, model: Union[nn.Module, nn.Sequential], device: Optional[torch.device] = None,
                     tokenizer: Optional[Callable] = None, max_len: Optional[int] = None,
                     batch_size: int = int(1e10), dtype: type = np.float32) -> Union[np.ndarray, torch.Tensor]:
    """
    Prediction function used for preprocessing step of drift detector.

    Parameters
    ----------
    x
        Batch of instances.
    model
        Model used for preprocessing.
    device
        Device type used. The default None tries to use the GPU and falls back on CPU if needed.
        Can be specified by passing either torch.device('cuda') or torch.device('cpu').
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
        return predict_batch(x, model, device=device, batch_size=batch_size, dtype=dtype)
    else:
        return predict_batch_transformer(x, model, tokenizer, max_len,
                                         device=device, batch_size=batch_size, dtype=dtype)
