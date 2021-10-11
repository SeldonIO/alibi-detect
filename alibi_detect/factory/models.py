from alibi_detect.factory.utils import instantiate_class
from alibi_detect.utils.saving import load_tf_model
from functools import partial
import os
import torch.nn as nn
from transformers import AutoTokenizer
from typing import Callable, Optional, Union
#from benchmark.saving import load_model  # TODO

MODEL_KWARGS = ['pretrained', 'load_path', 'model_kwargs']


def get_model(model_name: str, n_classes: Optional[int], pretrained: Optional[bool] = True,
              load_path: Optional[str] = None, **kwargs) -> Union[nn.Module, nn.Sequential]:

    # TODO - what do we want the "model" to be? Just use load_tf_model here with given filepath?
    try:
        model = instantiate_class(model_name)(**kwargs['model_kwargs'])
    except:  # TODO
        raise ValueError(f'{model_name} not supported.')

#    # TODO - below
#    if load_path is not None and os.path.isdir(load_path):  # optionally load model weights
#        print(f'Load {model_name} weights.')
#        model = load_model(model, load_path, model_name=model_name)
    return model


def get_model_kwargs(model_kwargs: dict) -> dict:
    kwargs = {}
    kwargs_list = list(model_kwargs.keys())
    for k in MODEL_KWARGS:
        if k in kwargs_list:
            kwargs.update({k: model_kwargs[k]})
    return kwargs


def get_tokenizer(model_name: str, max_length: int) -> Callable:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return partial(tokenizer, padding=True, truncation=True, max_length=max_length, return_tensors='pt')