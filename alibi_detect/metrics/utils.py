import json
import numpy as np
from creme.utils.histogram import Histogram
from typing import Callable


def map_nested_dicts(ob: dict, func: Callable) -> dict:
    if isinstance(ob, dict):
        return {k: map_nested_dicts(v, func) for k, v in ob.items()}
    else:
        return func(ob)


def get_creme_value(metric):
    if isinstance(metric, Histogram):
        return creme_hist_to_dict(metric)
    else:
        return metric.get()


def creme_hist_to_dict(hist: Histogram) -> dict:
    d = {}
    for index, bin in enumerate(hist):
        d[index] = {'left': bin.left, 'right': bin.right, 'count': bin.count}
    return d


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (
                np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
                np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
