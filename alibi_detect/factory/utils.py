from alibi_detect.models import custom_models
import logging
import os
from pathlib import Path
from typing import Callable, Union, Type, Optional
from ruamel.yaml import YAML
from importlib import import_module
import tensorflow as tf

logger = logging.getLogger(__name__)

# List of absolute paths of nested dict fields we want to resolve.
# NOTE: We could instead just allow any fields (i.e. search entire cfg dict for dict values beginning with @,
#  or any nested fields named 'model' etc.
# TODO - decide on this...
FIELDS_TO_RESOLVE = [
    ['preprocess', 'model', 'source']
]


def get_nested_value(dic, keys):
    for key in keys:
        try:
            dic = dic[key]
        except TypeError or KeyError:
            return None
    return dic


def set_nested_value(dic, keys, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value


def instantiate_class(module: str, name: str, *args, **kwargs) -> Type[Callable]:
    klass = getattr(import_module(module), name)
    if hasattr(klass, 'validate'):
        klass.validate(*args, **kwargs)
    return klass(*args, **kwargs)


def read_detector_config(path: Union[os.PathLike, str]) -> dict:
    """
    This function reads a detector yaml config file and returns a dict specifying the detector.
    """
    yaml = YAML(typ='safe')
    cfg = yaml.load(Path(path))
    logger.info('Loaded config file from %s', path)
    return cfg


def resolve_cfg(cfg: dict, verbose: Optional[bool] = False) -> dict:
    for key in FIELDS_TO_RESOLVE:
        src = get_nested_value(cfg, key)
        if src.startswith('@'):
            src = src[1:]
            if src in custom_models.get_all():
                model = custom_models.get(src)
                cfg['registries'].update({src: custom_models.find(src)})
            else:
                raise ValueError("Can't find %s in the custom model registry" % src)
            if verbose:
                logger.info('Successfully resolved registry entry %s' % src)
        # Download model from uri
        elif src.startswith('http'):
            tf.keras.utils.get_file('tmp.h5', src, cache_dir='.')  # TODO - need to handle non-tensorflow artefacts
            model = tf.keras.models.load_model('datasets/tmp.h5')  # TODO - need better warning messages here
            if verbose:
                logger.info('Successfully fetched tensorflow model from %s' % src)
        # Model loaded from local filepath
        elif Path(src).is_file():
            model = tf.keras.models.load_model(src)
            if verbose:
                logger.info('Successfully loaded tensorflow model from %s' % src)
        else:
            model = src

        # Put the resolved model into the cfg dict
        set_nested_value(cfg, key, model)

    return cfg
