
from alibi_detect.cd.pytorch import preprocess_drift as preprocess_drift_torch
from alibi_detect.cd.tensorflow import preprocess_drift as preprocess_drift_tf
from alibi_detect.factory.models import load_model  # get_tokenizer
from functools import partial
import logging
from torch import device as torch_device
# from alibi_detect.cd.pytorch import HiddenOutput  # TODO - add tensorflow support
# from alibi_detect.models.pytorch import TransformerEmbedding
# import torch.nn as nn
# from transformers.tokenization_utils_base import BatchEncoding
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# TODO - pytorch, graphs, text etc
def load_preprocessor(cfg: dict,
                      backend: Optional[str] = 'tensorflow',
                      verbose: Optional[bool] = False) -> Optional[Callable]:
    if 'type' in cfg:
        preprocessor_type = cfg.pop('type')
    else:
        raise ValueError('The preprocessor `type` must be specified')

    # Process processor_type specific arguments e.g. model and tokenizer
    if preprocessor_type == 'model':
        if 'model' in cfg:
            model_cfg = cfg.pop('model')
            model = load_model(model_cfg, backend=backend, verbose=verbose)
        else:
            raise ValueError("A `model` must be specified when `preprocessor_type='model'`")
    elif preprocessor_type == 'transformer_embedding':
        return None  # TODO
    else:
        logger.warning('No valid preprocessing type specified. No preprocessing function is defined.')
        return None

    # Process backend and optional kwargs
    kwargs = {}
    if backend == 'tensorflow':
        preprocess_drift = preprocess_drift_tf
    elif backend == 'pytorch':
        preprocess_drift = preprocess_drift_torch
        # kwarg: device (pytorch only)
        if 'device' in cfg:
            device = cfg.pop('device')
            device = torch_device(device)
            kwargs.update({'device': device})
    # kwarg: preprocess_batch_fn
    if 'preprocess_batch_fn' in cfg:
        preprocess_batch_fn = cfg.pop('preprocess_batch_fn')
        pass  # TODO
    # kwarg: tokenizer
    if 'tokenizer' in cfg:
        tokenizer = cfg.pop('tokenizer')
        pass  # TODO
    # Other optional kwargs
    remaining_kwargs = ['max_len', 'batch_size', 'dtype']
    for key in remaining_kwargs:
        if key in cfg:
            kwargs.update({key: cfg.pop(key)})

    # All remaining entries in cfg are ignored
    for key in cfg.keys():
        logger.warning('%s is not a recognised field for the preprocess_fn and will be ignored' % key)

    return partial(preprocess_drift, model=model, **kwargs)

# def text_batch(data: Union[Tuple[str], List[str]], tokenizer: Callable) \
#         -> BatchEncoding:
#     return tokenizer(list(data))


# def get_preprocess_batch_fn(data_type: str, model_name: str = None, max_length: int = None) -> Optional[Callable]:
# #    if data_type == 'graph':  # TODO
# #        return graph_batch
#     if data_type == 'text':
#         tokenizer = get_tokenizer(model_name, max_length)
#         return partial(text_batch, tokenizer=tokenizer)
#     else:
#         return None
