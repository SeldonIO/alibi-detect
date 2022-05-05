from abc import ABC, abstractmethod
import copy
import json
import numpy as np
from typing import Dict, Any, Optional
from alibi_detect.version import __version__, __config_spec__

DEFAULT_META = {
    "name": None,
    "detector_type": None,  # online or offline
    "data_type": None,  # tabular, image or time-series
    "version": None,
    "config_spec": None,
    "version_warning": False
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
        self.meta['config_spec'] = __config_spec__
        self.meta['version_warning'] = False

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


# "Large artefacts" - to save memory these are skipped in _set_config(), but added back in get_config()
LARGE_ARTEFACTS = ['x_ref', 'c_ref', 'preprocess_fn']


class DriftConfigMixin:
    """
    A mixin class containing methods related to a drift detector's configuration dictionary.
    """
    config: Optional[dict] = None

    def get_config(self) -> dict:  # TODO - move to BaseDetector once config save/load implemented for non-drift
        """
        Get the detector's configuration dictionary.

        Returns
        -------
        The detector's configuration dictionary.
        """
        if self.config is not None:
            # Get config (stored in top-level self)
            cfg = self.config
            # Get low-level nested detector (if needed)
            detector = self._detector if hasattr(self, '_detector') else self  # type: ignore[attr-defined]
            detector = detector._detector if hasattr(detector, '_detector') else detector  # type: ignore[attr-defined]
            # Add large artefacts back to config
            for key in LARGE_ARTEFACTS:
                if hasattr(detector, key):
                    cfg[key] = getattr(detector, key)
            # Set x_ref_preprocessed flag
            cfg['x_ref_preprocessed'] = detector.preprocess_at_init and detector.preprocess_fn is not None
            return cfg
        else:
            raise NotImplementedError('Getting a config (or saving via a config file) is not yet implemented for this'
                                      'detector')

    @classmethod
    def from_config(cls, config: dict):
        """
        Instantiate a drift detector from a fully resolved (and validated) config dictionary.

        Parameters
        ----------
        config
            A config dictionary matching the schema's in :class:`~alibi_detect.saving.schemas`.
        """
        meta = config.pop('meta', None)  # meta is pop'd as don't want to pass as arg/kwarg
        detector = cls(**config)
        if meta is not None:
            detector.meta['version_warning'] = meta.get('version_warning', False)  # type: ignore[attr-defined]
            detector.config['meta']['version_warning'] = meta.get('version_warning', False)
        return detector

    def _set_config(self, inputs):  # TODO - move to BaseDetector once config save/load implemented for non-drift
        if self.config is None:  # init config if it doesn't already exist
            name = self.__class__.__name__
            # strip off any backend suffix
            backends = ['TF', 'Torch', 'Sklearn']
            for backend in backends:
                if name.endswith(backend):
                    name = name[:-len(backend)]
            # Init config dict
            self.config: Dict[str, Any] = {
                'name': name,
                'meta': {
                    'version': self.meta['version'],
                    'config_spec': self.meta['config_spec'],
                    'version_warning': self.meta['version_warning'],
                }
            }

        # args and kwargs
        pop_inputs = ['self', '__class__', '__len__']
        pop_inputs += LARGE_ARTEFACTS  # # pop large artefacts and add back in get_config()
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
