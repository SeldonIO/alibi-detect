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

def resolve_config(obj, cfg, path):
    path = path.replace('/config.toml', '')
    for key in obj.FROM_PATH:
        name = cfg[key]
        print(name)
        if '.dill' in name:
            # cfg[key] = dill.load(open(f'{path}/{key}/{name}', 'rb'))
            cfg[key] = dill.load(open(f'{name}', 'rb'))
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
            cfg[key] = deserialize(val, f'{path}/{key}')

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