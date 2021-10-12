from alibi_detect.models import custom_models
from alibi_detect.cd.tensorflow import HiddenOutput
from pathlib import Path
import torch.nn as nn
import tensorflow as tf
from transformers import AutoTokenizer
from typing import Callable, Optional, Union
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


# TODO - Add pytorch functionality, and tokenizer etc
def load_model(orig_cfg: dict) \
        -> Union[nn.Module, nn.Sequential, tf.keras.Model]:

    cfg = deepcopy(orig_cfg)
    if 'source' in cfg:
        model_src = cfg.pop('source')
    else:
        raise ValueError('Model `source` not specified')

    # Custom registered models
    if model_src.startswith('@'):
        model_src = model_src[1:]
        if model_src in custom_models.get_all():
            model = custom_models.get(model_src)
        else:
            raise ValueError("Can't find %s in the custom model registry" % model_src)
    # Download model from uri
    elif model_src.startswith('http'):
        tf.keras.utils.get_file('tmp.h5', model_src, cache_dir='.')
        model = tf.keras.models.load_model('datasets/tmp.h5')
    # Model loaded from local filepath
    elif Path(model_src).is_file():
        model = tf.keras.models.load_model(model_src)
    else:
        raise ValueError('No valid model source found')

    # Extract layers if needed
    if 'type' in cfg:
        model_type = cfg.pop('type')
    else:
        raise ValueError('`model_type` must be specified along with any model')
    if model_type == 'hidden':
        model = HiddenOutput(model,
                             layer=cfg.pop('layer', -1),
                             flatten=cfg.pop('flatten', False)
                             )
    elif model_type == 'encoder':
        try:
            model = model.encoder
        except AttributeError:
            logger.warning('No encoder attribute found in model, assuming the model is already an encoder...')
    return model

# TODO
# def get_tokenizer(model_name: str, max_length: int) -> Callable:
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     return partial(tokenizer, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
