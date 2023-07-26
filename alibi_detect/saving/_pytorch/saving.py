import os
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import dill  # dispatch table setting not done here as done in top-level saving.py file
import torch
import torch.nn as nn

from alibi_detect.cd.pytorch import UAE, HiddenOutput
from alibi_detect.models.pytorch import TransformerEmbedding
from alibi_detect.utils.frameworks import Framework
from alibi_detect.utils._types import TorchDeviceType

logger = logging.getLogger(__name__)


def save_model_config(model: Callable,
                      base_path: Path,
                      local_path: Path = Path('.')) -> Tuple[dict, Optional[dict]]:
    """
    Save a PyTorch model to a config dictionary. When a model has a text embedding model contained within it,
    this is extracted and saved separately.

    Parameters
    ----------
    model
        The model to save.
    base_path
        Base filepath to save to (the location of the `config.toml` file).
    local_path
        A local (relative) filepath to append to base_path.

    Returns
    -------
    A tuple containing the model and embedding config dicts.
    """
    cfg_model: Optional[Dict[str, Any]] = None
    cfg_embed: Optional[Dict[str, Any]] = None
    if isinstance(model, UAE):
        layers = list(model.encoder.children())
        if isinstance(layers[0], TransformerEmbedding):  # if UAE contains embedding and encoder
            # embedding
            embed = layers[0]
            cfg_embed = save_embedding_config(embed, base_path, local_path.joinpath('embedding'))
            # preprocessing encoder
            model = layers[1]
        else:  # If UAE is simply an encoder
            model = model.encoder
    elif isinstance(model, TransformerEmbedding):
        cfg_embed = save_embedding_config(model, base_path, local_path.joinpath('embedding'))
        model = None
    elif isinstance(model, HiddenOutput):
        model = model.model
    elif isinstance(model, nn.Module):  # Last as TransformerEmbedding and UAE are nn.Module's
        model = model
    else:
        raise ValueError('Model not recognised, cannot save.')

    if model is not None:
        filepath = base_path.joinpath(local_path)
        save_model(model, filepath=filepath)
        cfg_model = {
            'flavour': Framework.PYTORCH.value,
            'src': local_path.joinpath('model')
        }
    return cfg_model, cfg_embed


def save_model(model: nn.Module,
               filepath: Union[str, os.PathLike],
               save_dir: Union[str, os.PathLike] = 'model') -> None:
    """
    Save PyTorch model.

    Parameters
    ----------
    model
        The PyTorch model to save.
    filepath
        Save directory.
    save_dir
        Name of folder to save to within the filepath directory.
    """
    # create folder to save model in
    model_path = Path(filepath).joinpath(save_dir)
    if not model_path.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(model_path))
        model_path.mkdir(parents=True, exist_ok=True)

    # save model
    model_path = model_path.joinpath('model.pt')

    if isinstance(model, nn.Module):
        torch.save(model, model_path, pickle_module=dill)
    else:
        raise ValueError('The extracted model to save is not a `nn.Module`. Cannot save.')


def save_embedding_config(embed: TransformerEmbedding,
                          base_path: Path,
                          local_path: Path = Path('.')) -> dict:
    """
    Save embeddings for text drift models.

    Parameters
    ----------
    embed
        Embedding model.
    base_path
        Base filepath to save to (the location of the `config.toml` file).
    local_path
        A local (relative) filepath to append to base_path.
    """
    # create folder to save model in
    filepath = base_path.joinpath(local_path)
    if not filepath.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(filepath))
        filepath.mkdir(parents=True, exist_ok=True)

    # Populate config dict
    cfg_embed: Dict[str, Any] = {}
    cfg_embed.update({'type': embed.emb_type})
    cfg_embed.update({'layers': embed.hs_emb.keywords['layers']})
    cfg_embed.update({'src': local_path})
    cfg_embed.update({'flavour': Framework.PYTORCH.value})

    # Save embedding model
    logger.info('Saving embedding model to {}.'.format(filepath))
    embed.model.save_pretrained(filepath)

    return cfg_embed


def save_device(device: TorchDeviceType):
    """

    Parameters
    ----------
    device
        Torch device to be serialised. Can be specified by passing either ``'cuda'``,
        ``'gpu'``, ``'cpu'`` or an instance of ``torch.device``.

    Returns
    -------
    a string with value ``'cuda'`` or ``'cpu'``.
    """
    device_str = str(device)
    return device_str.split(':')[0]
