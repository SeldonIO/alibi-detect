from alibi_detect.utils.missing_optional_dependency import import_optional

KNNTorch = import_optional('alibi_detect.od.backend.torch.knn', ['KNNTorch'])
PValNormalizerTorch, ShiftAndScaleNormalizerTorch, TopKAggregatorTorch, AverageAggregatorTorch, \
    MaxAggregatorTorch, MinAggregatorTorch, AccumulatorTorch = import_optional(
        'alibi_detect.od.backend.torch.ensemble',
        ['PValNormalizer', 'ShiftAndScaleNormalizer', 'TopKAggregator',
         'AverageAggregator', 'MaxAggregator', 'MinAggregator', 'Accumulator']
    )
