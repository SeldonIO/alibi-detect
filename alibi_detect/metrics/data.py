from creme.utils.histogram import Histogram
from creme import stats
import numpy as np
from typing import Union
from copy import deepcopy
import logging
import json

from alibi_detect.metrics.utils import map_nested_dicts, get_creme_value, NumpyEncoder

logger = logging.getLogger(__name__)

UNIVARIATE_STATS = {
    'mean': stats.Mean,
    'variance': stats.Var,
    'median': stats.Quantile
}

BIVARIATE_STATS = {
    # 'covariance': stats.Covariance # needs two inputs to update
}  # type: dict

HISTOGRAM = {
    'histogram': Histogram
}

NUMERICAL_METRICS = {**UNIVARIATE_STATS, **BIVARIATE_STATS, **HISTOGRAM}
CATEGORICAL_METRICS = {**HISTOGRAM}  # TODO: custom max bins?


class DataTracker:

    def __init__(self, n_features: int, cat_vars: dict = None) -> None:
        # set up metrics for numerical features only
        if cat_vars is None:
            cat_vars = {}
        categorical_cols = list(cat_vars.keys())
        numerical_cols = np.setdiff1d(range(n_features), categorical_cols).tolist()
        metrics = {}
        for col in numerical_cols:
            metrics[col] = deepcopy(NUMERICAL_METRICS)
            for key, val in metrics[col].items():
                metrics[col][key] = val()
        for col in categorical_cols:
            metrics[col] = deepcopy(CATEGORICAL_METRICS)
            for key, val in metrics[col].items():
                metrics[col][key] = val()

        self.metrics = metrics
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols

    def update(self, X: np.ndarray) -> None:
        # TODO: triple for loop...
        for x in X:
            for index, metrics in self.metrics.items():
                for name, metric in metrics.items():
                    metric.update(x[index])

    def get(self, serialize=True) -> Union[dict, str]:
        result = map_nested_dicts(self.metrics, get_creme_value)
        if serialize:
            return json.dumps(result, cls=NumpyEncoder)
        return result
