import numpy as np
from typing import Callable, Union


def tokenize_transformer(x: Union[list, np.ndarray], tokenizer: Callable, max_len: int, backend: str) -> dict:
    """
    Batch tokenizer for transformer models.

    Parameters
    ----------
    x
        Batch of instances.
    tokenizer
        Tokenizer for model.
    max_len
        Max token length.
    backend
        PyTorch ('pt') or TensorFlow ('tf') backend.

    Returns
    -------
    Dictionary with tokenized instances.
    """
    return tokenizer(list(x), pad_to_max_length=True, max_length=max_len, return_tensors=backend)
