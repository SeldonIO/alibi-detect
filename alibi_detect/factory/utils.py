"""
This submodule is based on utils.py from
https://github.com/SeldonIO/seldon-research/tree/master/alibi-explain-factory-config.
"""
import logging
import os
from pathlib import Path
from typing import Callable, Optional, Union, Tuple
from ruamel.yaml import YAML
from importlib import import_module

# TODO - use gin_config instead of this?
# from registry import REGISTRY

logger = logging.getLogger(__name__)


def instantiate_class(module: str, name: str, *args, **kwargs) -> Callable:
    klass = getattr(import_module(module), name)
    if hasattr(klass, 'validate'):
        klass.validate(*args, **kwargs)
    return klass(*args, **kwargs)


def read_detector_config(path: Union[os.PathLike, str]) -> dict:
    """
    This function reads a detector yaml config file and returns a dict specifying the detector.

    Any classes with pydantic schema's are validated before instantiating the class, and references are resolved.
    """
    yaml = YAML(typ='safe')
    cfg = yaml.load(Path(path))
    logger.info('Loaded config file from %s', path)
    # cfg = resolve_config(cfg) # TODO
    # logger.info('Successfully resolved references')
    return cfg


#def download_from_gs(uri: str) -> None:
#    tf.keras.utils.get_file('tmp.h5', uri, cache_dir='.')
#
#
#def download_from_gs_and_load_tf_model(uri: str) -> tf.keras.Model:
#    download_from_gs(uri)
#    return tf.keras.models.load_model('datasets/tmp.h5')
#
#
#def resolve_config(config: dict) -> dict:
#    """
#    This function parses the config and replaces `@string` keys with entries in the registry.
#    It also fetches persisted tf.keras .h5 models from cloud storage.
#    """
#    for key, val in config.items():
#        if isinstance(val, dict):
#            resolve_config(val)
#        elif isinstance(val, str):
##            if val.startswith('@'):  #TODO - either use REGISTRY or replace with gin_config
##                strkey = val[1:]
##                try:
##                    config[key] = REGISTRY[strkey]
##                except KeyError:
##                    logger.exception('No object found for string key %s', strkey)
##                    raise
#            if val.startswith('http'):
#                model = download_from_gs_and_load_tf_model(val)
#                config[key] = model
#            else:
#                pass
#        else:
#            pass
#    return config


#def validate_config(config: dict) -> dict:
#    """
#    This function validates the config against a pydantic model.
#    """
#    return AlibiConfig(**config).dict()
