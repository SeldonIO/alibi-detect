from functools import partial
import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Union
from alibi_detect.utils.prediction import tokenize_transformer


def predict_batch(x: Union[list, np.ndarray, torch.Tensor], model: Union[nn.Module, nn.Sequential],
                  device: torch.device = None, batch_size: int = int(1e10), preprocess_fn: Callable = None,
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
    preprocess_fn
        Optional preprocessing function for each batch.
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
    n = len(x)
    n_minibatch = int(np.ceil(n / batch_size))
    return_np = not isinstance(dtype, torch.dtype)
    preds = []
    with torch.no_grad():
        for i in range(n_minibatch):
            istart, istop = i * batch_size, min((i + 1) * batch_size, n)
            x_batch = x[istart:istop]
            if isinstance(preprocess_fn, Callable):  # type: ignore
                x_batch = preprocess_fn(x_batch)
            preds_tmp = model(x_batch.to(device))
            if device.type == 'cuda':
                preds_tmp = preds_tmp.cpu()
            if return_np:
                preds_tmp = preds_tmp.numpy()
            preds.append(preds_tmp)
    if return_np:
        return np.concatenate(preds)
    else:
        return torch.cat(preds)


def predict_batch_transformer(x: Union[list, np.ndarray], model: Union[nn.Module, nn.Sequential],
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
    preprocess_fn = partial(tokenize_transformer, tokenizer=tokenizer, max_len=max_len, backend='pt')
    return predict_batch(x, model, device=device, preprocess_fn=preprocess_fn, batch_size=batch_size, dtype=dtype)
