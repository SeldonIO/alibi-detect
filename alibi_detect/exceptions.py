"""This module defines the Alibi exception hierarchy and common exceptions used across the library."""

from abc import ABC


class AlibiDetectException(Exception, ABC):
    def __init__(self, message: str) -> None:
        """Abstract base class of all alibi detect exceptions.

        Parameters
        ----------
        message
            The error message.
        """
        super().__init__(message)


class NotFitException(AlibiDetectException):
    """Exception raised when a transform is not fitted."""

    pass


class ThresholdNotInferredException(AlibiDetectException):
    """Exception raised when a threshold not inferred for an outlier detector."""

    pass
