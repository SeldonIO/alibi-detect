import logging
import os
from pathlib import Path
from typing import Union
import toml
import dill

from alibi_detect.saving.registry import registry

logger = logging.getLogger(__name__)


def read_config(filepath: Union[os.PathLike, str]) -> dict:
    filepath = Path(filepath)
    cfg = dict(toml.load(filepath))  # toml.load types return as MutableMapping, force to dict
    logger.info('Loaded config file from {}'.format(str(filepath)))
    return cfg


def load_detector(path):
    if 'config.toml' not in str(path):
        path = str(path) + '/config.toml'
    config_file = Path(path)
    cfg = read_config(config_file)
    object_name = cfg.pop('name')
    object_meta = cfg.pop('meta')
    obj = registry.get_all()[object_name]
    detector = obj.deserialize(cfg)
    return detector
