from __future__ import annotations

from alibi_detect.version import __version__, __config_spec__
import logging
from typing import Dict, Any
from pathlib import Path
from typing import Union
import toml
import os

logger = logging.getLogger(__name__)

def write_config(cfg: dict, filepath: Union[str, os.PathLike]):
    filepath = Path(filepath)
    if not filepath.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(filepath))
        filepath.mkdir(parents=True, exist_ok=True)

    logger.info('Writing config to {}'.format(filepath.joinpath('config.toml')))
    with open(filepath.joinpath('config.toml'), 'w') as f:
        toml.dump(cfg, f, encoder=toml.TomlNumpyEncoder())  # type: ignore[misc]


class ConfigMixin:
    """Mixed in to anything you want to save.

    define CONFIG_PARAMS, LARGE_PARAMS and BASE_OBJ on the class you've mixed this into in order to
    obtain the desired behavour. Users should be able to use this mix in on there own objects in order
    to ensure they are serialisable. 
    """
    CONFIG_PARAMS: tuple = ()   # set of args passed to init that will be in config
    LARGE_PARAMS: tuple = ()    # set of args passed to init that are big and are added to config when it's getted
    BASE_OBJ: bool = False      # Base objects are things like detectors and Ensembles that should have there own 
                                # self contained config and be referenced from other configs.
    TO_STR: tuple = ()          # Function objects that need to be saved as `.dill` files or are contained in the 
                                # registry. If an attribute is added here then the save_detector function will
                                # save the attribute in config as a path.

    def _set_config(self, inputs):
        name = self.__class__.__name__
        config: Dict[str, Any] = {
            'name': name,
        }

        if self.BASE_OBJ:
            config['meta'] = {
                'version': __version__,
                'config_spec': __config_spec__,
            }

        for key in self.CONFIG_PARAMS:
            if key not in self.LARGE_PARAMS: config[key] = inputs[key]
        self.config = config

    def get_config(self) -> dict:
        if self.config is not None:
            cfg = self.config
            for key in self.LARGE_PARAMS:
                cfg[key] = getattr(self, key)
        else:
            raise NotImplementedError('Getting a config (or saving via a config file) is not yet implemented for this'
                                      'detector')

        return cfg

    @classmethod
    def from_config(cls, config: dict):
        """
        Note: For custom or complicated behavour this should be overidden
        """
        return cls(**config)

    def serialize_value(self, key, val, path):
        path = f'{path}/{key}'

        if hasattr(self, f'_{key}_serializer'):
            # if _key_serializer is defined on the class we use that to serialze the key.
            key_serialiser = getattr(self, f'_{key}_serializer')
            return key_serialiser(self, val, path)
        
        elif hasattr(self, f'_{type(val).__name__}_serializer'):
            # if _type_serializer is defined on the class we use that to serialze the value.
            type_serialiser = getattr(self, f'_{type(val).__name__}_serializer')
            return type_serialiser(self, val, path)

        elif hasattr(val, 'BASE_OBJ') and not getattr(val, 'BASE_OBJ'):
            # if val extends ConfigMixin but isn't a BASE_OBJ we serialize it using the 
            # serialize method defined on the object
            return val.serialize(path)

        elif hasattr(val, 'BASE_OBJ') and getattr(val, 'BASE_OBJ'):
            # if val extends ConfigMixin and is a BASE_OBJ we save and serialize it as 
            # the path
            return val.save(path)

        else:
            return val

    def _list_serializer(self, key, val, path):
        return [self.serialize_value(ind, item, path) for ind, item in enumerate(val)]

    def _dict_serializer(self, key, val, path):
        return {k: self.serialize_value(k, v, path) for k, v in val.items()}

    def serialize(self, path):
        cfg = self.get_config()
        for key, val in cfg.items():
            cfg[key] = self.serialize_value(key, val, path)
        return cfg

    def save(self, path):
        cfg = self.serialize(path)
        write_config(cfg, path)
        return path