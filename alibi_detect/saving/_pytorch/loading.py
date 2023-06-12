import logging
import os
from importlib import import_module
from pathlib import Path
from typing import Callable, Optional, Union, Type

import dill
import torch
import torch.nn as nn

from alibi_detect.cd.pytorch import UAE, HiddenOutput
from alibi_detect.cd.pytorch.preprocess import _Encoder
from alibi_detect.models.pytorch import TransformerEmbedding
from alibi_detect.utils.pytorch.kernels import DeepKernel

logger = logging.getLogger(__name__)


def load_model(filepath: Union[str, os.PathLike],
               layer: Optional[int] = None,
               ) -> nn.Module:
    """
    Load PyTorch model.

    Parameters
    ----------
    filepath
        Saved model filepath.
    layer
        Optional index of a hidden layer to extract. If not `None`, a
        :py:class:`~alibi_detect.cd.pytorch.HiddenOutput` model is returned.

    Returns
    -------
    Loaded model.
    """
    filepath = Path(filepath).joinpath('model.pt')
    model = torch.load(filepath, pickle_module=dill)
    # Optionally extract hidden layer
    if isinstance(layer, int):
        model = HiddenOutput(model, layer=layer)
    return model


def prep_model_and_emb(model: nn.Module, emb: Optional[TransformerEmbedding]) -> nn.Module:
    """
    Function to perform final preprocessing of model (and/or embedding) before it is passed to preprocess_drift.

    Parameters
    ----------
    model
        A compatible model.
    emb
        An optional text embedding model.

    Returns
    -------
    The final model ready to passed to preprocess_drift.
    """
    # Process model (and embedding)
    model = model.encoder if isinstance(model, UAE) else model  # This is to avoid nesting UAE's already a UAE
    if emb is not None:
        model = _Encoder(emb, mlp=model)
        model = UAE(encoder_net=model)
    return model


def load_kernel_config(cfg: dict) -> Callable:
    """
    Loads a kernel from a kernel config dict.

    Parameters
    ----------
    cfg
        A kernel config dict. (see pydantic schema's).

    Returns
    -------
    The kernel.
    """
    if 'src' in cfg:  # Standard kernel config
        kernel = cfg.pop('src')
        if hasattr(kernel, 'from_config'):
            kernel = kernel.from_config(cfg)

    elif 'proj' in cfg:  # DeepKernel config
        # Kernel a
        kernel_a = cfg['kernel_a']
        kernel_b = cfg['kernel_b']
        if kernel_a != 'rbf':
            cfg['kernel_a'] = load_kernel_config(kernel_a)
        if kernel_b != 'rbf':
            cfg['kernel_b'] = load_kernel_config(kernel_b)
        # Assemble deep kernel
        kernel = DeepKernel.from_config(cfg)

    else:
        raise ValueError('Unable to process kernel. The kernel config dict must either be a `KernelConfig` with a '
                         '`src` field, or a `DeepkernelConfig` with a `proj` field.)')
    return kernel


def load_optimizer(cfg: dict) -> Type[torch.optim.Optimizer]:
    """
    Imports a PyTorch torch.optim.Optimizer class from an optimizer config dict.

    Parameters
    ----------
    cfg
        The optimizer config dict.

    Returns
    -------
    The loaded optimizer class.
    """
    class_name = cfg.get('class_name')
    try:
        return getattr(import_module('torch.optim'), class_name)
    except AttributeError:
        raise ValueError(f"{class_name} is not a recognised optimizer in `torch.optim`.")


def load_embedding(src: str, embedding_type, layers) -> TransformerEmbedding:
    """
    Load a pre-trained PyTorch text embedding from a directory.
    See the `:py:class:~alibi_detect.models.pytorch.TransformerEmbedding` documentation for a
    full description of the `embedding_type` and `layers` kwargs.

    Parameters
    ----------
    src
        Name of or path to the model.
    embedding_type
       Type of embedding to extract. Needs to be one of pooler_output,
       last_hidden_state, hidden_state or hidden_state_cls.
    layers
        A list with int's referring to the hidden layers used to extract the embedding.

    Returns
    -------
    The loaded embedding.
    """
    emb = TransformerEmbedding(src, embedding_type=embedding_type, layers=layers)
    return emb
