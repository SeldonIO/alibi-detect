from abc import ABC, abstractmethod
import copy
import numpy as np
from typing import Dict

DEFAULT_DATA = {"instance_score": None}  # type: Dict
DEFAULT_META = {
    "name": None,
    "detector_type": None,  # online or offline
    "data_type": None  # tabular, image or time-series
}  # type: Dict


def outlier_prediction_dict():
    data = DEFAULT_DATA
    data['feature_score'] = None
    data['is_outlier'] = None
    return copy.deepcopy({"data": data, "meta": DEFAULT_META})


def adversarial_prediction_dict():
    data = DEFAULT_DATA
    data['is_adversarial'] = None
    return copy.deepcopy({"data": data, "meta": DEFAULT_META})


def adversarial_correction_dict():
    data = DEFAULT_DATA
    data['is_adversarial'] = None
    data['corrected'] = None
    data['no_defense'] = None
    data['defense'] = None
    return copy.deepcopy({"data": data, "meta": DEFAULT_META})


def concept_drift_dict():
    data = {
        "batch_score": None,
        "feature_score": None,
        "is_drift": None
    }
    return copy.deepcopy({"data": data, "meta": DEFAULT_META})


class BaseDetector(ABC):
    """ Base class for outlier detection algorithms. """

    def __init__(self):
        self.meta = copy.deepcopy(DEFAULT_META)
        self.meta['name'] = self.__class__.__name__

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
