
from __future__ import annotations
from alibi_detect.version import __version__, __config_spec__
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


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
            return cfg
        else:
            raise NotImplementedError('Getting a config (or saving via a config file) is not yet implemented for this'
                                      'detector')


    @classmethod
    def from_config(cls, config: dict):
        """
        Note: For custom or complicated behavour this should be overidden
        """
        return cls(**config)
