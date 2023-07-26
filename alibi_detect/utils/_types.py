"""
Defining types compatible with different Python versions and defining custom types.
"""
from sklearn.base import BaseEstimator  # import here (instead of later) since sklearn currently a core dep
from alibi_detect.utils.frameworks import has_tensorflow, has_pytorch
from typing import Union, Type, Optional


# Literal for typing
from typing_extensions import Literal
from typing_extensions import TypeAlias


# Optional dep dependent tuples of types, for isinstance checks and pydantic
supported_models_tf: tuple = ()
supported_models_torch: tuple = ()
supported_optimizers_tf: tuple = ()
supported_optimizers_torch: tuple = ()
if has_tensorflow:
    import tensorflow as tf
    supported_models_tf = (tf.keras.Model, )
    if hasattr(tf.keras.optimizers, 'legacy'):
        supported_optimizers_tf = (tf.keras.optimizers.Optimizer, tf.keras.optimizers.legacy.Optimizer, type)
    else:
        supported_optimizers_tf = (tf.keras.optimizers.Optimizer, type)
if has_pytorch:
    import torch
    supported_models_torch = (torch.nn.Module, )
    supported_optimizers_torch = (type, )  # Note type not object!
supported_models_sklearn = (BaseEstimator, )
supported_models_all = supported_models_tf + supported_models_torch + supported_models_sklearn
supported_optimizers_all = supported_optimizers_tf + supported_optimizers_torch

# type aliases, for use with mypy (must be FwdRef's if involving opt. deps.)
OptimizerTF: TypeAlias = Union['tf.keras.optimizers.Optimizer', 'tf.keras.optimizers.legacy.Optimizer',
                               Type['tf.keras.optimizers.Optimizer'], Type['tf.keras.optimizers.legacy.Optimizer']]

TorchDeviceType: TypeAlias = Optional[Union[Literal['cuda', 'gpu', 'cpu'], 'torch.device']]
