from alibi_detect.cd.tensorflow import HiddenOutput as HiddenOutput_tf
from alibi_detect.cd.pytorch import HiddenOutput as HiddenOutput_torch
import torch.nn as nn
import tensorflow as tf
from transformers import AutoTokenizer
from typing import Callable, Optional, Union
import logging

logger = logging.getLogger(__name__)


# TODO - tokenizer etc
def load_model(cfg: dict,
               backend: Optional[str] = 'tensorflow',
               verbose: Optional[bool] = False) \
        -> Union[nn.Module, nn.Sequential, tf.keras.Model]:

    if 'source' in cfg:
        model_src = cfg.pop('source')
    else:
        raise ValueError('Model `source` not specified')

    # Check if model is still wrapped within a function (i.e. a resolved registry function)
    if callable(model_src) and not isinstance(model_src, (nn.Module, nn.Sequential, tf.keras.Model)):
        model = model_src()
    else:
        model = model_src

    # Check if model is compatible now
    if not isinstance(model, (nn.Module, nn.Sequential, tf.keras.Model)):
        raise ValueError('The specified model is not a compatible tensorflow or pytorch model.')

    # Extract layers if needed #TODO - capability to define custom layers
    if 'type' in cfg:
        model_type = cfg.pop('type')
    else:
        raise ValueError('`model_type` must be specified along with any model')
    if model_type == 'hidden':
        if backend == 'tensorflow':
            HiddenOutput = HiddenOutput_tf
        else:
            HiddenOutput = HiddenOutput_torch
        model = HiddenOutput(model,
                             layer=cfg.pop('layer', -1),
                             flatten=cfg.pop('flatten', False))
    elif model_type == 'encoder':
        try:
            model = model.encoder
        except AttributeError:
            if verbose:
                logger.warning('No encoder attribute found in model, assuming the model is already an encoder...')
    return model

# TODO
# def get_tokenizer(model_name: str, max_length: int) -> Callable:
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     return partial(tokenizer, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
