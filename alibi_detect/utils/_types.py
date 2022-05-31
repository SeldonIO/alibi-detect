"""
Defining types compatible with different Python versions and defining custom types.
"""
import sys
from typing import Any, Generic, Optional, Type, TypeVar
import numpy as np
from numpy.lib import NumpyVersion
from pydantic.fields import ModelField

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
