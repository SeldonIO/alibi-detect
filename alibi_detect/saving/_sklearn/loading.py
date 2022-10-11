import os
from pathlib import Path
from typing import Union

import joblib
from sklearn.base import BaseEstimator


def load_model(filepath: Union[str, os.PathLike],
               ) -> BaseEstimator:
    """
    Load scikit-learn (or xgboost) model. Models are assumed to be a subclass of :class:`~sklearn.base.BaseEstimator`.
    This includes xgboost models following the scikit-learn API
    (see https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn).

    Parameters
    ----------
    filepath
        Saved model directory.

    Returns
    -------
    Loaded model.
    """
    model_dir = Path(filepath)
    return joblib.load(model_dir.joinpath('model.joblib'))
