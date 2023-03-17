"""
Defining types compatible with different Python versions and defining custom types.
"""
import sys
from sklearn.base import BaseEstimator  # import here (instead of later) since sklearn currently a core dep
from alibi_detect.utils.frameworks import has_tensorflow, has_pytorch

# Literal for typing
if sys.version_info >= (3, 8):
    from typing import Literal  # noqa
else:
    from typing_extensions import Literal  # noqa


# Optional dep dependent tuples of types
supported_models_tf: tuple = ()
supported_models_torch: tuple = ()
supported_optimizers_tf: tuple = ()
supported_optimizers_torch: tuple = ()
if has_tensorflow:
    import tensorflow as tf
    supported_models_tf = (tf.keras.Model, )
    supported_optimizers_tf = (tf.keras.optimizers.Optimizer, type)
if has_pytorch:
    import torch
    supported_models_torch = (torch.nn.Module, )
    supported_optimizers_torch = (type, )  # Note type not object!
supported_models_sklearn = (BaseEstimator, )
supported_models_all = supported_models_tf + supported_models_torch + supported_models_sklearn
supported_optimizers_all = supported_optimizers_tf + supported_optimizers_torch
