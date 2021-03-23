from functools import partial
import logging
import numpy as np
import tensorflow as tf
from typing import Callable, Dict, Optional, Tuple, Union
from alibi_detect.base import BaseDetector, concept_drift_dict

logger = logging.getLogger(__name__)


class MarginDensityDrift(BaseDetector):

    def __init__(self,
                 margin: float = None,
                 model: Union[tf.keras.Model, tf.keras.Sequential] = None,
                 density_range: Tuple = None,
                 preprocess_fn: Optional[Callable] = None,
                 preprocess_kwargs: Optional[dict] = None,
                 data_type: Optional[str] = None
                 ) -> None:
        """
        Margin density concept drift detector based on the following paper:
        https://www.researchgate.net/publication/282542797_Don't_Pay_for_Validation_Detecting_Drifts_from_Unlabeled_data_Using_Margin_Density

        Parameters
        ----------
        margin
            Width of margin at decision boundary.
        model
            A trained tf.keras binary classification model.
        density_range
            Tuple of length 2 that defines margin density lower and upper bounds.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        preprocess_kwargs
            Kwargs for `preprocess_fn`.
        data_type
            Optionally specify the data type (tabular or image). Added to metadata.
        """
        super().__init__()

        if margin is None:
            logger.warning('Need to set margin to detect data drift.')
        elif density_range is None:
            logger.warning('Need to set density_range to detect data drift.')

        if isinstance(preprocess_fn, Callable) and isinstance(preprocess_kwargs, dict):  # type: ignore
            self.preprocess_fn = partial(preprocess_fn, **preprocess_kwargs)
        else:
            self.preprocess_fn = preprocess_fn  # type: ignore

        self.margin = margin
        self.model = model
        self.density_range = density_range

        # set metadata
        self.meta['detector_type'] = 'offline'
        self.meta['data_type'] = data_type

    def preprocess(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Data preprocessing before computing the drift scores.

        Parameters
        ----------
        X
            Batch of instances.

        Returns
        -------
        Preprocessed new instances.
        """
        if isinstance(self.preprocess_fn, Callable):  # type: ignore
            X = self.preprocess_fn(X)
        return X

    def score(self, X: Union[np.ndarray, list]) -> float:
        """
        Compute the margin density for a batch of production data.

        Parameters
        ----------
        X
            Batch of instances.

        Returns
        -------
        Margin density.
        """
        X = self.preprocess(X)

        preds = self.model.predict(X).ravel()
        class_prob_diff = abs(preds*2-1)
        num_in_margin_preds = sum(class_prob_diff <= self.margin)
        num_total_preds = len(class_prob_diff)
        margin_density = num_in_margin_preds / num_total_preds

        return margin_density

    def predict(self, X: Union[np.ndarray, list]) \
            -> Dict[Dict[str, str], Dict[str, Union[int, float]]]:
        """
        Predict whether a batch of data has drifted from the reference data.

        Parameters
        ----------
        X
            Batch of instances.

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the drift prediction, margin, margin_density and density_range.
        """
        # compute drift scores
        margin_density = self.score(X)

        # populate drift dict
        cd = concept_drift_dict()
        cd['meta'] = self.meta
        if margin_density < self.density_range[0]:
            cd['data']['direction'] = 'low margin density'
            cd['data']['is_drift'] = 1
        elif margin_density > self.density_range[1]:
            cd['data']['direction'] = 'high margin density'
            cd['data']['is_drift'] = 1
        else:
            cd['data']['direction'] = None
            cd['data']['is_drift'] = 0
        cd['data']['margin'] = self.margin
        cd['data']['margin_density'] = margin_density
        cd['data']['density_range'] = self.density_range
        return cd
