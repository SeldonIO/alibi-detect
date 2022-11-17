import pytest
import torch
from alibi_detect.od.backend.torch import ensemble


def test_pval_normaliser():
    normaliser = ensemble.PValNormaliser()
    x = torch.randn(3, 10)
    x_ref = torch.randn(64, 10)
    with pytest.raises(ValueError):
        normaliser(x)

    normaliser.fit(x_ref)
    x_norm = normaliser(x)
    normaliser = torch.jit.script(normaliser)
    x_norm_2 = normaliser(x)
    assert torch.all(x_norm_2 == x_norm)


def test_shift_and_scale_normaliser():
    normaliser = ensemble.ShiftAndScaleNormaliser()
    x = torch.randn(3, 10)
    x_ref = torch.randn(64, 10)
    with pytest.raises(ValueError):
        normaliser(x)

    normaliser.fit(x_ref)
    x_norm = normaliser(x)
    normaliser = torch.jit.script(normaliser)
    x_norm_2 = normaliser(x)
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
@pytest.mark.parametrize('normaliser', ['PValNormaliser', 'ShiftAndScaleNormaliser'])
def test_accumulator(aggregator, normaliser):
    aggregator = getattr(ensemble, aggregator)()
    normaliser = getattr(ensemble, normaliser)()
    accumulator = ensemble.Accumulator(aggregator=aggregator, normaliser=normaliser)

    x = torch.randn(3, 10)
    x_ref = torch.randn(64, 10)

    accumulator.fit(x_ref)
    x_norm = accumulator(x)
    accumulator = torch.jit.script(accumulator)
    x_norm_2 = accumulator(x)
    assert torch.all(x_norm_2 == x_norm)