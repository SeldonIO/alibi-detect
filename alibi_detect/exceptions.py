"""This module defines the Alibi Detect exception hierarchy and common exceptions used across the library."""
from typing_extensions import Literal
from typing import Callable
from abc import ABC
from functools import wraps


class AlibiDetectException(Exception, ABC):
    def __init__(self, message: str) -> None:
        """Abstract base class of all alibi detect errors.

        Parameters
        ----------
        message
            The error message.
        """
        super().__init__(message)


class NotFittedError(AlibiDetectException):
    def __init__(self, object_name: str) -> None:
        """Exception raised when a transform is not fitted.

        Parameters
        ----------
        object_name
            The name of the unfit object.
        """
        message = f'{object_name} has not been fit!'
        super().__init__(message)


class ThresholdNotInferredError(AlibiDetectException):
    def __init__(self, object_name: str) -> None:
        """Exception raised when a threshold not inferred for an outlier detector.

        Parameters
        ----------
        object_name
            The name of the object that does not have a threshold fit.
        """
        message = f'{object_name} has no threshold set, call `infer_threshold` to fit one!'
        super().__init__(message)


def _catch_error(err_name: Literal['NotFittedError', 'ThresholdNotInferredError']) -> Callable:
    """Decorator to catch errors and raise a more informative error message.

    Note: This decorator should only be used on detector frontend methods. It catches errors raised by
    backend components and re-raises them with error messages corresponding to the specific detector frontend.
    This is done to avoid exposing the backend components to the user.
    """
    error_type = globals()[err_name]

    def decorate(f):
        @wraps(f)
        def applicator(self, *args, **kwargs):
            try:
                return f(self, *args, **kwargs)
            except error_type as err:
                raise error_type(self.__class__.__name__) from err
        return applicator
    return decorate
