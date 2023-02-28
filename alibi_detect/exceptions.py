"""This module defines the Alibi exception hierarchy and common exceptions used across the library."""

from abc import ABC


class AlibiDetectError(Exception, ABC):
    def __init__(self, message: str) -> None:
        """Abstract base class of all alibi detect errors.

        Parameters
        ----------
        message
            The error message.
        """
        super().__init__(message)


class NotFittedError(AlibiDetectError):
    """Exception raised when a transform is not fitted."""

    pass


class ThresholdNotInferredError(AlibiDetectError):
    """Exception raised when a threshold not inferred for an outlier detector."""

    pass
