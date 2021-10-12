"""
This submodule is based on utils.py from
https://github.com/SeldonIO/seldon-research/tree/master/alibi-explain-factory-config.
"""
import logging
import os
from pathlib import Path
from typing import Callable, Union, Type
from ruamel.yaml import YAML
from importlib import import_module

logger = logging.getLogger(__name__)


def instantiate_class(module: str, name: str, *args, **kwargs) -> Type[Callable]:
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
    # cfg = resolve_config(cfg) # NOTE - could resolve @ and http references here instead (see builder.py NOTE)
    # logger.info('Successfully resolved references')
    return cfg
