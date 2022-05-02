from abc import ABC, abstractmethod
import copy
import json
import numpy as np
from typing import Dict, Any, Optional, Callable
from alibi_detect.version import __version__, __config_spec__

DEFAULT_META = {
    "name": None,
    "detector_type": None,  # online or offline
    "data_type": None,  # tabular, image or time-series
    "version": None,
}  # type: Dict


def outlier_prediction_dict():
    data = {
        'instance_score': None,
        'feature_score': None,
        'is_outlier': None
    }
    return copy.deepcopy({"data": data, "meta": DEFAULT_META})


def adversarial_prediction_dict():
    data = {
        'instance_score': None,
        'is_adversarial': None
    }
    return copy.deepcopy({"data": data, "meta": DEFAULT_META})


def adversarial_correction_dict():
    data = {
        'instance_score': None,
        'is_adversarial': None,
        'corrected': None,
        'no_defense': None,
        'defense': None
    }
    return copy.deepcopy({"data": data, "meta": DEFAULT_META})


def concept_drift_dict():
    data = {
        'is_drift': None,
        'distance': None,
        'p_val': None,
        'threshold': None
    }
    return copy.deepcopy({"data": data, "meta": DEFAULT_META})


class BaseDetector(ABC):
    """ Base class for outlier, adversarial and drift detection algorithms. """

    def __init__(self):
        self.meta = copy.deepcopy(DEFAULT_META)
        self.meta['name'] = self.__class__.__name__
        self.meta['version'] = __version__

    def __repr__(self):
        return self.__class__.__name__

    @property
    def meta(self) -> Dict:
        return self._meta

    @meta.setter
    def meta(self, value: Dict):
        if not isinstance(value, dict):
            raise TypeError('meta must be a dictionary')
        self._meta = value

    @abstractmethod
    def score(self, X: np.ndarray):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray):
        pass


class FitMixin(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        pass


class ThresholdMixin(ABC):
    @abstractmethod
    def infer_threshold(self, X: np.ndarray) -> None:
        pass


class DriftConfigMixin:
    """
    A mixin class to be used by detector `get_config` methods. The `drift_config` method defines the initial
    generic configuration dict for all detectors, which is then fully populated by a detector's get_config method(s).
    """
    x_ref: np.ndarray
    enable_config: bool = True

    def get_config(self) -> dict:  # TODO - move to BaseDetector once config save/load implemented for non-drift
        """
        Get the detector's configuration dictionary.

        Returns
        -------
        The detector's configuration dictionary.
        """
        if self.enable_config:
            return self.config
        else:
            raise ValueError('The detector must be instantiated with `enable_config=True` in order for the '
                             '`get_config` method to be used.')

    def _set_config(self, inputs):  # TODO - move to BaseDetector once config save/load implemented for non-drift
        if not hasattr(self, 'config'):  # init config if it doesn't already exist
            name = self.__class__.__name__
            # strip off any backend suffix
            backends = ['TF', 'Torch', 'Sklearn']
            for backend in backends:
                if name.endswith(backend):
                    name = name[:-len(backend)]
            # Init config dict
            cfg: Dict[str, Any] = {'name': name}

            # Populate meta dict and add to config
            cfg_meta = {
                'version': __version__,
                'config_spec': __config_spec__,
                'version_warning': self.meta.get('version_warning', False)
            }
            cfg.update({'meta': cfg_meta})
            self.config = cfg

        # args and kwargs
        pop_inputs = ['self', '__class__']
        pop_inputs += self.config.keys()  # Adding self.config.keys() avoids overwriting existing config
        [inputs.pop(k, None) for k in pop_inputs]
        self.config.update(inputs)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(
                obj,
                (
                        np.int_,
                        np.intc,
                        np.intp,
                        np.int8,
                        np.int16,
                        np.int32,
                        np.int64,
                        np.uint8,
                        np.uint16,
                        np.uint32,
                        np.uint64,
                ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
