from alibi_detect.utils.missing_optional_dependency import import_optional
from alibi_detect.utils._types import Literal
from alibi_detect.od.base import TransformProtocol, transform_protocols
from typing import Union


KNNTorch = import_optional('alibi_detect.od.backend.torch.knn', ['KNNTorch'])
PValNormalizerTorch, ShiftAndScaleNormalizerTorch, TopKAggregatorTorch, AverageAggregatorTorch, \
    MaxAggregatorTorch, MinAggregatorTorch, AccumulatorTorch = import_optional(
        'alibi_detect.od.backend.torch.ensemble',
        ['PValNormalizer', 'ShiftAndScaleNormalizer', 'TopKAggregator',
         'AverageAggregator', 'MaxAggregator', 'MinAggregator', 'Accumulator']
    )

normalizer_literals = Literal['PValNormalizerTorch', 'ShiftAndScaleNormalizerTorch']
aggregator_literals = Literal['TopKAggregatorTorch', 'AverageAggregatorTorch',
                              'MaxAggregatorTorch', 'MinAggregatorTorch']


def get_normalizer(normalizer: Union[transform_protocols, normalizer_literals]) -> TransformProtocol:
    if isinstance(normalizer, str):
        try:
            return {
                'PValNormalizerTorch': PValNormalizerTorch,
                'ShiftAndScaleNormalizerTorch': ShiftAndScaleNormalizerTorch,
            }.get(normalizer, )()
        except KeyError:
            raise NotImplementedError(f'Normalizer {normalizer} not implemented.')
    return normalizer


def get_aggregator(aggregator: Union[TransformProtocol, aggregator_literals]) -> TransformProtocol:
    if isinstance(aggregator, str):
        try:
            return {
                'TopKAggregatorTorch': TopKAggregatorTorch,
                'AverageAggregatorTorch': AverageAggregatorTorch,
                'MaxAggregatorTorch': MaxAggregatorTorch,
                'MinAggregatorTorch': MinAggregatorTorch,
            }.get(aggregator, )()
        except KeyError:
            raise NotImplementedError(f'Aggregator {aggregator} not implemented.')
    return aggregator
