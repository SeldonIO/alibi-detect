import logging
import os
from pathlib import Path
from typing import Union
import toml
import dill

from alibi_detect.saving.registry import registry
from alibi_detect.saving.saving import _serialize_object

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
            cfg_path = save_detector(item, path)
            return cfg_path
        if not item.BASE_OBJ:
            cfg = item.get_config()

            for key in item.TO_STR:
                path, _ = _serialize_object(
                    cfg[key], base_path=Path(path), 
                    local_path=Path(key)
                )
                cfg[key] = path

            return cfg
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
    # cfg = dictmap(cfg, curry_serialize_objs(path))
    write_config(cfg, path)
    return path + '/config.toml'


def read_config(filepath: Union[os.PathLike, str]) -> dict:
    filepath = Path(filepath)
    cfg = dict(toml.load(filepath))  # toml.load types return as MutableMapping, force to dict
    logger.info('Loaded config file from {}'.format(str(filepath)))
    return cfg


def resolve_config(obj, cfg, path):
    path = path.replace('/config.toml', '')
    for key in obj.TO_STR:
        name = cfg[key]
        if '.dill' in name:
            cfg[key] = dill.load(open(f'{path}/{name}', 'rb'))
        if name.startswith('@'):
            name = name[1:]
            if name in registry.get_all():
                cfg[key] = registry.get(name)
    return cfg

def deserialize(item, path):
    if isinstance(item, str) and 'config.toml' in item:
        return load_detector(item)
    if isinstance(item, dict) and item.get('name'):
        obj_name = item.pop('name')
        obj = registry.get_all()[obj_name]
        item = resolve_config(obj, item, path)
        obj_instance = obj.from_config(item)
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
    version_warning = meta.pop('version_warning', False)
    obj = registry.get_all()[obj_name]
    cfg = resolve_config(obj, cfg, path)
    obj_instance = obj.from_config(cfg)
    obj_instance.meta = meta
    obj_instance.meta['version_warning'] = version_warning
    obj_instance.config['meta']['version_warning'] = version_warning
    return obj_instance
