from alibi_detect.utils.missing_optional_dependency import import_optional

from typing import Union
from typing_extensions import Literal, Protocol, runtime_checkable


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


@runtime_checkable
class FittedTransformProtocol(TransformProtocol, Protocol):
    """Protocol for fitted transformer objects.

    This protocol models the joint interface of the :py:obj:`~alibi_detect.od.pytorch.ensemble.BaseTransformTorch`
    class and the :py:obj:`~alibi_detect.od.pytorch.ensemble.FitMixinTorch` class. These objects are transforms that
    require to be fit."""
    def fit(self, x_ref):
        pass

    def set_fitted(self):
        pass

    def check_fitted(self):
        pass


TransformProtocolType = Union[TransformProtocol, FittedTransformProtocol]
NormalizerLiterals = Literal['PValNormalizer', 'ShiftAndScaleNormalizer']
AggregatorLiterals = Literal['TopKAggregator', 'AverageAggregator',
                             'MaxAggregator', 'MinAggregator']


PValNormalizer, ShiftAndScaleNormalizer, TopKAggregator, AverageAggregator, \
    MaxAggregator, MinAggregator = import_optional(
        'alibi_detect.od.pytorch.ensemble',
        ['PValNormalizer', 'ShiftAndScaleNormalizer', 'TopKAggregator',
         'AverageAggregator', 'MaxAggregator', 'MinAggregator']
    )


def get_normalizer(normalizer: Union[TransformProtocolType, NormalizerLiterals]) -> TransformProtocol:
    if isinstance(normalizer, str):
        try:
            return {
                'PValNormalizer': PValNormalizer,
                'ShiftAndScaleNormalizer': ShiftAndScaleNormalizer,
            }.get(normalizer)()
        except KeyError:
            raise NotImplementedError(f'Normalizer {normalizer} not implemented.')
    return normalizer


def get_aggregator(aggregator: Union[TransformProtocol, AggregatorLiterals]) -> TransformProtocol:
    if isinstance(aggregator, str):
        try:
            return {
                'TopKAggregator': TopKAggregator,
                'AverageAggregator': AverageAggregator,
                'MaxAggregator': MaxAggregator,
                'MinAggregator': MinAggregator,
            }.get(aggregator)()
        except KeyError:
            raise NotImplementedError(f'Aggregator {aggregator} not implemented.')
    return aggregator
