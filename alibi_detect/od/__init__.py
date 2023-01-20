from alibi_detect.utils.missing_optional_dependency import import_optional

from .isolationforest import IForest
from .mahalanobis import Mahalanobis
from .sr import SpectralResidual

from alibi_detect.od.base import TransformProtocol, transform_protocols
from typing_extensions import Literal
from typing import Union

PValNormalizer, ShiftAndScaleNormalizer, TopKAggregator, AverageAggregator, \
    MaxAggregator, MinAggregator = import_optional(
        'alibi_detect.od.pytorch.ensemble',
        ['PValNormalizer', 'ShiftAndScaleNormalizer', 'TopKAggregator',
         'AverageAggregator', 'MaxAggregator', 'MinAggregator']
    )


normalizer_literals = Literal['PValNormalizer', 'ShiftAndScaleNormalizer']
aggregator_literals = Literal['TopKAggregator', 'AverageAggregator',
                              'MaxAggregator', 'MinAggregator']


def get_normalizer(normalizer: Union[transform_protocols, normalizer_literals]) -> TransformProtocol:
    if isinstance(normalizer, str):
        try:
            return {
                'PValNormalizer': PValNormalizer,
                'ShiftAndScaleNormalizer': ShiftAndScaleNormalizer,
            }.get(normalizer)()
        except KeyError:
            raise NotImplementedError(f'Normalizer {normalizer} not implemented.')
    return normalizer


def get_aggregator(aggregator: Union[TransformProtocol, aggregator_literals]) -> TransformProtocol:
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


OutlierAEGMM = import_optional('alibi_detect.od.aegmm', names=['OutlierAEGMM'])
OutlierAE = import_optional('alibi_detect.od.ae', names=['OutlierAE'])
OutlierVAE = import_optional('alibi_detect.od.vae', names=['OutlierVAE'])
OutlierVAEGMM = import_optional('alibi_detect.od.vaegmm', names=['OutlierVAEGMM'])
OutlierSeq2Seq = import_optional('alibi_detect.od.seq2seq', names=['OutlierSeq2Seq'])
LLR = import_optional('alibi_detect.od.llr', names=['LLR'])
OutlierProphet = import_optional('alibi_detect.od.prophet', names=['OutlierProphet'])
KNN = import_optional('alibi_detect.od.knn', names=['KNN'])

__all__ = [
    "OutlierAEGMM",
    "IForest",
    "Mahalanobis",
    "OutlierAE",
    "OutlierVAE",
    "OutlierVAEGMM",
    "OutlierSeq2Seq",
    "SpectralResidual",
    "LLR",
    "OutlierProphet"
    "KNN"
    "PValNormalizer",
    "ShiftAndScaleNormalizer",
    "TopKAggregator",
    "AverageAggregator",
    "MaxAggregator",
    "MinAggregator",
]
