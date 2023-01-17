import sys
from typing import Any, Generic, Optional, Type, TypeVar, Union, List
import numpy as np
from numpy.lib import NumpyVersion
from pydantic.fields import ModelField

from alibi_detect.utils.frameworks import has_tensorflow, has_pytorch, has_keops, Framework

if has_tensorflow:
    import tensorflow as tf
if has_pytorch:
    import torch


def coerce_int2list(value: int) -> List[int]:
    """Validator to coerce int to list (pydantic doesn't do this by default)."""
    if not isinstance(value, list):
        return [value]
    else:
        return value


# Framework validator (validates `flavour` and `backend` fields)
def validate_framework(framework: str, field: ModelField) -> str:
    if (framework == Framework.TENSORFLOW and has_tensorflow) or (framework == Framework.PYTORCH and has_pytorch) or \
            (framework == Framework.KEOPS and has_keops):
        return framework
    elif framework == Framework.SKLEARN:  # sklearn is a core dep
        return framework
    else:
        raise ImportError(f"`{field.name} = '{framework}'` not possible since {framework} is not installed.")


# NumPy NDArray pydantic validator type
# The code below is adapted from https://github.com/cheind/pydantic-numpy.
T = TypeVar("T", bound=np.generic)
if NumpyVersion(np.__version__) < "1.22.0" or sys.version_info < (3, 9):
    class NDArray(Generic[T], np.ndarray):
        """
        A Generic pydantic model to coerce to np.ndarray's.
        """
        @classmethod
        def __get_validators__(cls):
            yield cls.validate

        @classmethod
        def validate(cls, val: Any, field: ModelField) -> np.ndarray:
            return _coerce_2_ndarray(cls, val, field)

else:
    class NDArray(Generic[T], np.ndarray[Any, T]):  # type: ignore[no-redef, type-var]
        """
        A Generic pydantic model to coerce to np.ndarray's.
        """
        @classmethod
        def __get_validators__(cls):
            yield cls.validate

        @classmethod
        def validate(cls, val: Any, field: ModelField) -> Optional[np.ndarray]:
            return _coerce_2_ndarray(cls, val, field)


def _coerce_2_ndarray(cls: Type, val: Any, field: ModelField) -> np.ndarray:
    if field.sub_fields is not None:
        dtype_field = field.sub_fields[0]
        return np.asarray(val, dtype=dtype_field.type_)
    else:
        return np.asarray(val)


def coerce_2_tensor(value: Union[float, List[float]], values: dict):
    if value is None:
        return value
    framework = values.get('backend') or values.get('flavour')
    if framework is None:
        raise ValueError('`coerce_2tensor` failed since no framework identified.')
    elif framework == Framework.TENSORFLOW and has_tensorflow:
        return tf.convert_to_tensor(value)
    elif (framework == Framework.PYTORCH and has_pytorch) or (framework == Framework.KEOPS and has_keops):
        return torch.tensor(value)
    else:
        # Error should not be raised since `flavour` should have already been validated.
        raise ImportError(f'Cannot coerce to {framework} Tensor since {framework} is not installed.')
