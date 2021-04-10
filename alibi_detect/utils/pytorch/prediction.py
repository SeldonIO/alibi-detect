import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Union


def predict_batch(x: Union[np.ndarray, torch.Tensor], model: Union[nn.Module, nn.Sequential],
                  device: torch.device = None, batch_size: int = int(1e10),
                  dtype: Union[np.float32, torch.dtype] = np.float32) -> Union[np.ndarray, torch.Tensor]:
    """
    Make batch predictions on a model.

    Parameters
    ----------
    x
        Batch of instances.
    model
        PyTorch model.
    device
        Device type used. The default None tries to use the GPU and falls back on CPU if needed.
        Can be specified by passing either torch.device('cuda') or torch.device('cpu').
    batch_size
        Batch size used during prediction.
    dtype
        Model output type, e.g. np.float32 or torch.float32.

    Returns
    -------
    Numpy array or torch tensor with model outputs.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    n = x.size(0)
    n_minibatch = int(np.ceil(n / batch_size))
    return_np = not isinstance(dtype, torch.dtype)
    preds = []
    with torch.no_grad():
        for i in range(n_minibatch):
            istart, istop = i * batch_size, min((i + 1) * batch_size, n)
            preds_tmp = model(x[istart:istop].to(device))
            if device.type == 'cuda':
                preds_tmp = preds_tmp.cpu()
            if return_np:
                preds_tmp = preds_tmp.numpy()
            preds.append(preds_tmp)
    if return_np:
        return np.concatenate(preds)
    else:
        return torch.cat(preds)


def predict_batch_transformer(x: Union[np.ndarray, torch.Tensor], model: Union[nn.Module, nn.Sequential],
                              tokenizer: Callable, max_len: int, device: torch.device = None,
                              batch_size: int = int(1e10), dtype: Union[np.float32, torch.dtype] = np.float32) \
        -> Union[np.ndarray, torch.Tensor]:
    """
    Make batch predictions using a transformers tokenizer and model.

    Parameters
    ----------
    x
        Batch of instances.
    model
        PyTorch model.
    tokenizer
        Tokenizer for model.
    max_len
        Max sequence length for tokens.
    device
        Device type used. The default None tries to use the GPU and falls back on CPU if needed.
        Can be specified by passing either torch.device('cuda') or torch.device('cpu').
    batch_size
        Batch size used during prediction.
    dtype
        Model output type, e.g. np.float32 or torch.float32.

    Returns
    -------
    Numpy array or torch tensor with model outputs.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n = x.shape[0]
    n_minibatch = int(np.ceil(n / batch_size))
    return_np = not isinstance(dtype, torch.dtype)
    preds = []
    with torch.no_grad():
        for i in range(n_minibatch):
            istart, istop = i * batch_size, min((i + 1) * batch_size, n)
            tokens = tokenizer.batch_encode_plus(  # type: ignore
                x[istart:istop], pad_to_max_length=True, max_length=max_len, return_tensors='pt'
            ).to(device)
            preds_tmp = model(tokens)
            if device.type == 'cuda':
                preds_tmp = preds_tmp.cpu()
            if return_np:
                preds_tmp = preds_tmp.numpy()
            preds.append(preds_tmp)
    if return_np:
        return np.concatenate(preds)
    else:
        return torch.cat(preds)
