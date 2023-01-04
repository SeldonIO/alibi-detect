import pytest
import torch

from alibi_detect.od.backend.torch import ensemble
from alibi_detect.od.base import NotFitException


def test_pval_normalizer():
    normalizer = ensemble.PValNormalizer()
    x = torch.randn(3, 10)
    x_ref = torch.randn(64, 10)
    # unfit normalizer raises exception
    with pytest.raises(NotFitException) as err:
        normalizer(x)
    assert err.value.args[0] == 'PValNormalizer has not been fit!'

    normalizer.fit(x_ref)
    x_norm = normalizer(x)
    normalizer = torch.jit.script(normalizer)
    x_norm_2 = normalizer(x)
    assert torch.all(x_norm_2 == x_norm)


def test_shift_and_scale_normalizer():
    normalizer = ensemble.ShiftAndScaleNormalizer()
    x = torch.randn(3, 10)
    x_ref = torch.randn(64, 10)
    # unfit normalizer raises exception
    with pytest.raises(NotFitException) as err:
        normalizer(x)
    assert err.value.args[0] == 'ShiftAndScaleNormalizer has not been fit!'

    normalizer.fit(x_ref)
    x_norm = normalizer(x)
    normalizer = torch.jit.script(normalizer)
    x_norm_2 = normalizer(x)
    assert torch.all(x_norm_2 == x_norm)


def test_average_aggregator():
    aggregator = ensemble.AverageAggregator()
    scores = torch.randn((3, 10))
    aggregated_scores = aggregator(scores)
    assert aggregated_scores.shape == (3, )
    aggregator = torch.jit.script(aggregator)
    aggregated_scores_2 = aggregator(scores)
    assert torch.all(aggregated_scores_2 == aggregated_scores)


def test_weighted_average_aggregator():
    aggregator = ensemble.AverageAggregator(weights=torch.randn((10)))
    scores = torch.randn((3, 10))
    aggregated_scores = aggregator(scores)
    assert aggregated_scores.shape == (3, )
    aggregator = torch.jit.script(aggregator)
    aggregated_scores_2 = aggregator(scores)
    assert torch.all(aggregated_scores_2 == aggregated_scores)


def test_topk_aggregator():
    aggregator = ensemble.TopKAggregator(k=4)
    scores = torch.randn((3, 10))
    aggregated_scores = aggregator(scores)
    assert aggregated_scores.shape == (3, )
    aggregator = torch.jit.script(aggregator)
    aggregated_scores_2 = aggregator(scores)
    assert torch.all(aggregated_scores_2 == aggregated_scores)


def test_max_aggregator():
    aggregator = ensemble.MaxAggregator()
    scores = torch.randn((3, 10))
    aggregated_scores = aggregator(scores)
    assert aggregated_scores.shape == (3, )
    aggregator = torch.jit.script(aggregator)
    aggregated_scores_2 = aggregator(scores)
    assert torch.all(aggregated_scores_2 == aggregated_scores)


def test_min_aggregator():
    aggregator = ensemble.MinAggregator()
    scores = torch.randn((3, 10))
    aggregated_scores = aggregator(scores)
    assert aggregated_scores.shape == (3, )
    aggregator = torch.jit.script(aggregator)
    aggregated_scores_2 = aggregator(scores)
    assert torch.all(aggregated_scores_2 == aggregated_scores)


@pytest.mark.parametrize('aggregator', ['AverageAggregator', 'MaxAggregator', 'MinAggregator', 'TopKAggregator'])
@pytest.mark.parametrize('normalizer', ['PValNormalizer', 'ShiftAndScaleNormalizer'])
def test_accumulator(aggregator, normalizer):
    aggregator = getattr(ensemble, aggregator)()
    normalizer = getattr(ensemble, normalizer)()
    accumulator = ensemble.Accumulator(aggregator=aggregator, normalizer=normalizer)

    x = torch.randn(3, 10)
    x_ref = torch.randn(64, 10)

    accumulator.fit(x_ref)
    x_norm = accumulator(x)
    accumulator = torch.jit.script(accumulator)
    x_norm_2 = accumulator(x)
    assert torch.all(x_norm_2 == x_norm)
