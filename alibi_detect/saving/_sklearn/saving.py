import logging
import os
from pathlib import Path
from typing import Union
import joblib
from sklearn.base import BaseEstimator

from alibi_detect.utils.frameworks import Framework

logger = logging.getLogger(__name__)


def save_model_config(model: BaseEstimator,
                      base_path: Path,
                      local_path: Path = Path('.')) -> dict:
    """
    Save a scikit-learn (or xgboost) model to a config dictionary.
    Models are assumed to be a subclass of :class:`~sklearn.base.BaseEstimator`. This includes xgboost models
    following the scikit-learn API
    (see https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn).

    Parameters
    ----------
    model
        The model to save.
    base_path
        Base filepath to save to (the location of the `config.toml` file).
    local_path
        A local (relative) filepath to append to base_path.

    Returns
    -------
    The model config dict.
    """
    filepath = base_path.joinpath(local_path)
    save_model(model, filepath=filepath, save_dir='model')
    cfg_model = {
        'flavour': Framework.SKLEARN.value,
        'src': local_path.joinpath('model')
    }
    return cfg_model


def save_model(model: BaseEstimator,
               filepath: Union[str, os.PathLike],
               save_dir: Union[str, os.PathLike] = 'model') -> None:
    """
    Save scikit-learn (and xgboost) models. Models are assumed to be a subclass of :class:`~sklearn.base.BaseEstimator`.
    This includes xgboost models following the scikit-learn API
    (see https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn).

    Parameters
    ----------
    model
        The tf.keras.Model to save.
    filepath
        Save directory.
    save_dir
        Name of folder to save to within the filepath directory.
    """
    # create folder to save model in
    model_path = Path(filepath).joinpath(save_dir)
    if not model_path.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(model_path))
        model_path.mkdir(parents=True, exist_ok=True)

    # save model
    model_path = model_path.joinpath('model.joblib')
    joblib.dump(model, model_path)
