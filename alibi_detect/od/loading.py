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
    config_file = Path(path)
    cfg = read_config(config_file)
    object_name = cfg.pop('name')
    object_meta = cfg.pop('meta')
    obj = registry.get_all()[object_name]
    detector = obj.deserialize(cfg)
    return detector


    # print('cfg', cfg)

    # for key, val in cfg.items():
    #     if isinstance(val, (list, tuple)):
    #         cfg[key] = []
    #         for ind, item in enumerate(val):
    #             cfg[key].append(deserialize(item, f'{path}/{key}/{ind}'))
    #     else:
    #         cfg[key] = deserialize(val, f'{path}/{key}')

    # obj_name = cfg.pop('name')
    # meta = cfg.pop('meta')
    # version_warning = meta.pop('version_warning', False)
    # obj = registry.get_all()[obj_name]
    # cfg = resolve_config(obj, cfg, path)
    # obj_instance = obj.from_config(cfg)
    # obj_instance.meta = meta
    # obj_instance.meta['version_warning'] = version_warning
    # obj_instance.config['meta']['version_warning'] = version_warning
    # return obj_instance