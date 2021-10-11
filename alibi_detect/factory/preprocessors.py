from alibi_detect.cd.pytorch import HiddenOutput
from alibi_detect.models.pytorch import TransformerEmbedding
from alibi_detect.cd.pytorch import preprocess_drift as preprocess_drift_torch
from alibi_detect.cd.tensorflow import preprocess_drift as preprocess_drift_tf
from alibi_detect.factory.models import load_model, get_tokenizer
from functools import partial
import logging
from torch import device as torch_device
import torch.nn as nn  # TODO - add tensorflow support
from transformers.tokenization_utils_base import BatchEncoding
from typing import Callable, List, Optional, Tuple, Union
# from custom.models import Flatten # TODO

logger = logging.getLogger(__name__)


def init_preprocessor(cfg: dict, backend: Optional[str] = 'tensorflow') -> Optional[Callable]:
    if 'type' in cfg:
        preprocessor_type = cfg.pop('type')
    else:
        raise ValueError('The preprocessor `type` must be specified')

    # Process processor_type specific arguments e.g. model and tokenizer
    if preprocessor_type == 'model':
        if 'model' in cfg:
            model_cfg = cfg.pop('model')
            model = load_model(model_cfg)
        else:
            raise ValueError("A `model` must be specified when `preprocessor_type='model'`")

    elif preprocessor_type == 'custom':
        pass  # TODO

    elif preprocessor_type == 'transformer_embedding':
        pass  # TODO

    else:
        logger.warning('No valid preprocessing type specified. No preprocessing function is defined.')
        return None

    # Process batch size (optional)
    kwargs = {}
    if 'batch_size' in cfg:
        kwargs.update({'batch_size': cfg.pop('batch_size')})

    # Process backend specific arguments
    if backend=='tensorflow':
        preprocess_drift = preprocess_drift_tf

    elif backend=='pytorch':
        preprocess_drift = preprocess_drift_torch
        kwargs.update({'device': device})
    else:
        raise ValueError("`Backend` should be 'tensorflow' or 'pytorch'")

    return partial(preprocess_drift, model, **kwargs)

                   #device=device,
                   #preprocess_fn=preprocess_batch_fn, batch_size=batch_size)


#    if preprocessor_type == 'hidden':
#        model = HiddenOutput(orig_model, layer=kwargs['layer'], flatten=True)
#    elif preprocessor_type == 'encoder':
#        preprocess_model_name = kwargs.pop('model')
#        load_path = None if 'load_path' not in list(kwargs.keys()) else kwargs.pop('load_path')
#        model = get_model(preprocess_model_name, None, load_path=load_path, **kwargs).to(device)
#        if hasattr(model, 'encoder'):
#            model = model.encoder  # extract encoder from autoencoder
#        model = model.eval()
##    elif preprocess_type == 'flatten':  #TODO
##        model = Flatten()
#    elif preprocessor_type == 'model':
#        model = orig_model
#    elif preprocessor_type == 'transformerembedding':
#        model = TransformerEmbedding(kwargs['model_name'], kwargs['embedding_type'], kwargs['layers']).to(device).eval()
#    else:
#        try:
#            model = import_by_name(preprocess_type)
#        except:  # TODO
#            logger.warning('No valid preprocessing step identified. No preprocessing function defined.')
#            return None
#    return partial(predict_batch, model=model, device=device, preprocess_fn=preprocess_batch_fn, batch_size=batch_size)


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
