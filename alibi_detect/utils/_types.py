"""
Defining types compatible with different Python versions and defining custom types.
"""
import sys
from typing import Any, Generic, Optional, Type, TypeVar, Union
import numpy as np
from numpy.lib import NumpyVersion
from pydantic.fields import ModelField
from sklearn.base import BaseEstimator  # import here (instead of later) since sklearn currently a core dep
from alibi_detect.utils.frameworks import has_tensorflow, has_pytorch

# Literal for typing
if sys.version_info >= (3, 8):
    from typing import Literal  # noqa
else:
    from typing_extensions import Literal  # noqa

# NumPy NDArray pydantic validator type
# The code below is adapted from the great https://github.com/cheind/pydantic-numpy repo.
# TODO - above repo also has some very clever functionality to allow passing the numpy array as a npy/npz file.
#  This could be useful to explore in order to simplify our loading submodule.
T = TypeVar("T", bound=np.generic)
if NumpyVersion(np.__version__) < "1.22.0" or sys.version_info < (3, 9):
    class NDArray(Generic[T], np.ndarray):
        """
        A Generic pydantic model to validate (and coerce) np.ndarray's.
        """
        @classmethod
        def __get_validators__(cls):
            yield cls.validate

        @classmethod
        def validate(cls, val: Any, field: ModelField) -> np.ndarray:
            return _validate(cls, val, field)

else:
    class NDArray(Generic[T], np.ndarray[Any, T]):  # type: ignore[no-redef, type-var]
        """
        A Generic pydantic model to validate (and coerce) np.ndarray's.
        """
        @classmethod
        def __get_validators__(cls):
            yield cls.validate

        @classmethod
        def validate(cls, val: Any, field: ModelField) -> Optional[np.ndarray]:
            return _validate(cls, val, field)


def _validate(cls: Type, val: Any, field: ModelField) -> np.ndarray:
    if field.sub_fields is not None:
        dtype_field = field.sub_fields[0]
        return np.asarray(val, dtype=dtype_field.type_)
    else:
        return np.asarray(val)


# Optional dep dependent tuples of types
supported_models_tf, supported_models_torch, supported_models_sklearn = (), (), ()  # type: ignore
supported_optimizers_tf, supported_optimizers_torch = (), ()  # type: ignore
if has_tensorflow:
    import tensorflow as tf
    supported_models_tf = (tf.keras.Model, )  # type: ignore
    supported_optimizers_tf = (tf.keras.optimizers.Optimizer, )  # type: ignore
if has_pytorch:
    import torch
    supported_models_torch = (torch.nn.Module, torch.nn.Sequential)  # type: ignore
    supported_optimizers_torch = (torch.optim.Optimizer, )  # type: ignore
supported_models_sklearn = (BaseEstimator, )  # type: ignore
supported_models_all = supported_models_tf + supported_models_torch + supported_models_sklearn
supported_optimizers_all = supported_optimizers_tf + supported_optimizers_torch

# 2. Type unions
model_types_tf = Type['tf.keras.Model']
model_types_torch = Union['torch.nn.Module', 'torch.nn.Sequential']
model_types_sklearn = Type[BaseEstimator]  # no ForwardRef since sklearn a core dep
optimizer_types_tf = Type['tf.keras.optimizers.Optimizer']
optimizer_types_torch = Union['torch.optim.Optimizer']
optimizer_types_sklearn = Type[BaseEstimator]  # no ForwardRef since sklearn a core dep
model_types_all = Union[model_types_tf, model_types_torch, model_types_sklearn]
optimizer_types_all = Union[optimizer_types_tf, optimizer_types_torch]
