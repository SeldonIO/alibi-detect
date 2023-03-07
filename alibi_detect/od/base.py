from typing import Union

from typing_extensions import Protocol, runtime_checkable


# Use Protocols instead of base classes for the backend associated objects. This is a bit more flexible and allows us to
# avoid the torch/tensorflow imports in the base class.
@runtime_checkable
class TransformProtocol(Protocol):
    """Protocol for transformer objects.

    The :py:obj:`~alibi_detect.od.pytorch.ensemble.BaseTransformTorch` object provides abstract methods for
    objects that map between `torch` tensors. This protocol models the interface of the `BaseTransformTorch`
    class.
    """
    def transform(self, x):
        pass

    def _transform(self, x):
        pass


@runtime_checkable
class FittedTransformProtocol(TransformProtocol, Protocol):
    """Protocol for fitted transformer objects.

    The :py:obj:`~alibi_detect.od.pytorch.ensemble.BaseFittedTransformTorch` object provides abstract methods for
    objects that map between `torch` tensors and also require to be fit. This protocol models the interface of
    the `BaseFittedTransformTorch`
    class."""
    def fit(self, x_ref):
        pass

    def _fit(self, x_ref):
        pass

    def check_fitted(self):
        pass


transform_protocols = Union[TransformProtocol, FittedTransformProtocol]
