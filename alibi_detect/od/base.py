
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from alibi_detect.version import __version__, __config_spec__
import logging
from typing import Dict, Any, Optional
from alibi_detect.base import BaseDetector
from alibi_detect.saving.registry import registry

logger = logging.getLogger(__name__)


class ConfigMixin:
    def _set_config(self, inputs):
        # Set config metadata
        name = self.__class__.__name__

        # Init config dict
        self.config: Dict[str, Any] = {
            'name': name,
            'meta': {
                'version': __version__,
                'config_spec': __config_spec__,
            }
        }

        # args and kwargs
        pop_inputs = ['self', '__class__', '__len__', 'name', 'meta']
        [inputs.pop(k, None) for k in pop_inputs]

        for key, value in inputs.items():
            if hasattr(value, 'get_config'):
                inputs[key] = value.get_config()

        self.config.update(inputs)

    def get_config(self) -> dict:
        if self.config is not None:
            return self.config
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
        name = config.pop('name', False)

        # Init detector
        for key, val in config.items():
            if isinstance(val, dict) and val.get('meta'):
                sub_obj_name = val.pop('name', False)
                sub_obj_meta = val.pop('meta', None)
                ObjCls = registry.get_all()[sub_obj_name]
                obj = ObjCls(**val)
                config[key] = obj

        detector = cls(**config)
        # Add version_warning
        # detector.meta['version_warning'] = version_warning  # type: ignore[attr-defined]
        # detector.config['meta']['version_warning'] = version_warning
        return detector


class OutlierDetector(BaseDetector, ConfigMixin, ABC):
    """ Base class for outlier detection algorithms. """

    ensemble = False
    named_ensemble = False
    threshold_inferred = False


    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        pass


    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        pass


    def infer_threshold(self, X: np.ndarray, fpr: float) -> None:
        """
        Infers the threshold above which only fpr% of inlying data scores.
        Also saves down the scores to be later used for computing p-values
            of new data points (by comparison to the empirical cdf).
        For ensemble models the scores are normalised and aggregated before
            saving scores and inferring threshold.
        """
        self.val_scores = self.score(X)
        self.val_scores = self.normaliser.fit(self.val_scores).transform(self.val_scores) \
            if getattr(self, 'normaliser') else self.val_scores
        self.val_scores = self.aggregator.fit(self.val_scores).transform(self.val_scores) \
            if getattr(self, 'aggregator') else self.val_scores
        self.threshold = np.quantile(self.val_scores, 1-fpr)
        self.threshold_inferred = True


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Scores the instances and then compares to pre-inferred threshold.
        For ensemble models the scores from each constituent is added to the output.
        p-values are also returned by comparison to validation scores (of inliers)
        """
        output = {}
        scores = self.score(X)
        output['raw_scores'] = scores

        if getattr(self, 'normaliser') and self.normaliser.fitted:
            scores = self.normaliser.transform(scores)
            output['normalised_scores'] = scores

        if getattr(self, 'aggregator') and self.aggregator.fitted:
            scores = self.aggregator.transform(scores)
            output['aggregate_scores'] = scores

        if self.threshold_inferred:
            p_vals = (1 + (scores[:, None] < self.val_scores).sum(-1))/len(self.val_scores)
            preds = scores > self.threshold
            output.update(scores=scores, preds=preds, p_vals=p_vals)

        return output
