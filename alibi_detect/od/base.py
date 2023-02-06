from __future__ import annotations
from typing import Union

from typing_extensions import Protocol, runtime_checkable

# Use Protocols instead of base classes for the backend associated objects. This is a bit more flexible and allows us to
# avoid the torch/tensorflow imports in the base class.
@runtime_checkable
class TransformProtocol(Protocol):
    """Protocol for transformer objects."""
    def transform(self, x):
        pass

    def _transform(self, x):
        pass


@runtime_checkable
class FittedTransformProtocol(TransformProtocol, Protocol):
    """Protocol for fitted transformer objects."""
    def fit(self, x_ref):
        pass

    def _fit(self, x_ref):
        pass

    def check_fitted(self):
        pass


transform_protocols = Union[TransformProtocol, FittedTransformProtocol]
