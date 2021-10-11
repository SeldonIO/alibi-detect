from alibi_detect.cd.pytorch import HiddenOutput
from alibi_detect.models.pytorch import TransformerEmbedding
from alibi_detect.utils.pytorch import predict_batch
from alibi_detect.factory.utils import instantiate_class
from alibi_detect.factory.models import get_model, get_tokenizer
from functools import partial
import logging
from torch import device as torch_device
import torch.nn as nn  # TODO - add tensorflow support
from transformers.tokenization_utils_base import BatchEncoding
from typing import Callable, List, Optional, Tuple, Union
# from custom.models import Flatten # TODO

logger = logging.getLogger(__name__)


def init_preprocessor(cfg: dict) -> Callable:
    preprocessor_fn = get_preprocess_fn()
    return preprocessor_fn


def get_preprocess_fn(preprocess_type: str, orig_model: Optional[Union[nn.Module, nn.Sequential]],
                      device: torch_device, batch_size: int, preprocess_batch_fn: Optional[Callable], **kwargs) \
        -> Optional[Callable]:
    if preprocess_type == 'hidden':
        model = HiddenOutput(orig_model, layer=kwargs['layer'], flatten=True)
    elif preprocess_type == 'encoder':
        preprocess_model_name = kwargs.pop('model')
        load_path = None if 'load_path' not in list(kwargs.keys()) else kwargs.pop('load_path')
        model = get_model(preprocess_model_name, None, load_path=load_path, **kwargs).to(device)
        if hasattr(model, 'encoder'):
            model = model.encoder  # extract encoder from autoencoder
        model = model.eval()
#    elif preprocess_type == 'flatten':  #TODO
#        model = Flatten()
    elif preprocess_type == 'model':
        model = orig_model
    elif preprocess_type == 'transformerembedding':
        model = TransformerEmbedding(kwargs['model_name'], kwargs['embedding_type'], kwargs['layers']).to(device).eval()
    else:
        try:
            model = import_by_name(preprocess_type)
        except:  # TODO
            logger.warning('No valid preprocessing step identified. No preprocessing function defined.')
            return None
    return partial(predict_batch, model=model, device=device, preprocess_fn=preprocess_batch_fn, batch_size=batch_size)


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
