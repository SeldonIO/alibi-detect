from abc import ABC, abstractmethod
import os
import copy
import json
import numpy as np
from typing import Dict, Any, Optional, Union
from typing_extensions import Protocol, runtime_checkable
from alibi_detect.version import __version__


DEFAULT_META: Dict = {
    "name": None,
    "online": None,  # true or false
    "data_type": None,  # tabular, image or time-series
    "version": None,
    "detector_type": None  # drift, outlier or adversarial
}


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
    """Base class for outlier, adversarial and drift detection algorithms."""

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
    def fit(self, *args, **kwargs) -> None:
        ...


class ThresholdMixin(ABC):
    @abstractmethod
    def infer_threshold(self, *args, **kwargs) -> None:
        ...


# "Large artefacts" - to save memory these are skipped in _set_config(), but added back in get_config()
# Note: The current implementation assumes the artefact is stored as a class attribute, and as a config field under
# the same name. Refactoring will be required if this assumption is to be broken.
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
            # Add large artefacts back to config
            for key in LARGE_ARTEFACTS:
                if key in cfg and hasattr(self._nested_detector, key):
                    cfg[key] = getattr(self._nested_detector, key)
            # Set x_ref_preprocessed flag
            # If no preprocess_at_init, always true!
            preprocess_at_init = getattr(self._nested_detector, 'preprocess_at_init', True)
            cfg['x_ref_preprocessed'] = preprocess_at_init and self._nested_detector.preprocess_fn is not None
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
        # Check for existing version_warning. meta is pop'd as don't want to pass as arg/kwarg
        meta = config.pop('meta', None)
        meta = {} if meta is None else meta  # Needed because pydantic sets meta=None if it is missing from the config
        version_warning = meta.pop('version_warning', False)
        # Init detector
        detector = cls(**config)
        # Add version_warning
        detector.meta['version_warning'] = version_warning  # type: ignore[attr-defined]
        detector.config['meta']['version_warning'] = version_warning
        return detector

    def _set_config(self, inputs: dict):  # TODO - move to BaseDetector once config save/load implemented for non-drift
        """
        Set a detectors `config` attribute upon detector instantiation.

        Large artefacts are overwritten with `None` in order to avoid memory duplication. They're added back into
        the config later on by `get_config()`.

        Parameters
        ----------
        inputs
            The inputs (args/kwargs) given to the detector at instantiation.
        """
        # Set config metadata
        name = self.__class__.__name__

        # Init config dict
        self.config = {
            'name': name,
            'meta': {
                'version': __version__,
            }
        }

        # args and kwargs
        pop_inputs = ['self', '__class__', '__len__', 'name', 'meta']
        [inputs.pop(k, None) for k in pop_inputs]

        # Overwrite any large artefacts with None to save memory. They'll be added back by get_config()
        for key in LARGE_ARTEFACTS:
            if key in inputs and hasattr(self._nested_detector, key):
                inputs[key] = None

        self.config.update(inputs)

    @property
    def _nested_detector(self):
        """
        The low-level nested detector.
        """
        detector = self._detector if hasattr(self, '_detector') else self
        detector = detector._detector if hasattr(detector, '_detector') else detector
        return detector


@runtime_checkable
class Detector(Protocol):
    """Type Protocol for all detectors.

    Used for typing legacy save and load functionality in `alibi_detect.saving._tensorflow.saving.py`.

    Note
    ----
        This exists to distinguish between detectors with and without support for config saving and loading. Once all
        detector support this then this protocol will be removed.
    """

    meta: Dict

    def predict(self) -> Any: ...


@runtime_checkable
class ConfigurableDetector(Detector, Protocol):
    """Type Protocol for detectors that have support for saving via config.

    Used for typing save and load functionality in `alibi_detect.saving.saving`.
    """

    def get_config(self) -> dict: ...

    @classmethod
    def from_config(cls, config: dict): ...

    def _set_config(self, inputs: dict): ...


@runtime_checkable
class StatefulDetectorOnline(ConfigurableDetector, Protocol):
    """Type Protocol for detectors that have support for save/loading of online state.

    Used for typing save and load functionality in `alibi_detect.saving.saving`.
    """

    t: int = 0

    def save_state(self, filepath: Union[str, os.PathLike]): ...

    def load_state(self, filepath: Union[str, os.PathLike]): ...


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
