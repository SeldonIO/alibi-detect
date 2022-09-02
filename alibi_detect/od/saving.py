import logging
import os
from pathlib import Path
from typing import Union
import toml
from alibi_detect.saving.registry import registry

# from alibi_detect.saving.loading import _replace, validate_config

logger = logging.getLogger(__name__)


def write_config(cfg: dict, filepath: Union[str, os.PathLike]):
    filepath = Path(filepath)
    if not filepath.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(filepath))
        filepath.mkdir(parents=True, exist_ok=True)

    logger.info('Writing config to {}'.format(filepath.joinpath('config.toml')))
    with open(filepath.joinpath('config.toml'), 'w') as f:
        toml.dump(cfg, f, encoder=toml.TomlNumpyEncoder())  # type: ignore[misc]

def serialize(item, path):
    if hasattr(item, 'BASE_OBJ'):
        if item.BASE_OBJ:
            cfg_path = save_detector(item, path + '')
            return cfg_path
        if not item.BASE_OBJ:
            return item.get_config()
    else:
        return item

def save_detector(detector, path):
    cfg = detector.get_config()
    for key, val in cfg.items():
        if isinstance(val, (list, tuple)):
            cfg[key] = []
            for ind, item in enumerate(val):
                cfg[key].append(serialize(item, f'{path}/{key}/{ind}'))
        else:
            cfg[key] = serialize(val, path)
    write_config(cfg, path)
    return path + '/config.toml'


def read_config(filepath: Union[os.PathLike, str]) -> dict:
    filepath = Path(filepath)
    cfg = dict(toml.load(filepath))  # toml.load types return as MutableMapping, force to dict
    logger.info('Loaded config file from {}'.format(str(filepath)))
    return cfg


def deserialize(item, path):
    if isinstance(item, str) and 'config.toml' in item:
        return load_detector(item)
    if isinstance(item, dict) and item.get('name'):
        obj_name = item.pop('name')
        obj = registry.get_all()[obj_name]
        print(item, obj_name)
        obj_instance = obj(**item) # from config
        return obj_instance
    else:
        return item


def load_detector(path):
    config_file = Path(path)
    cfg = read_config(config_file)
    
    for key, val in cfg.items():
        if isinstance(val, (list, tuple)):
            cfg[key] = []
            for ind, item in enumerate(val):
                cfg[key].append(deserialize(item, f'{path}/{key}/{ind}'))
        else:
            cfg[key] = deserialize(val, path)

    obj_name = cfg.pop('name')
    meta = cfg.pop('meta')
    obj = registry.get_all()[obj_name]
    obj = obj(**cfg)
    obj.meta = meta
    return obj