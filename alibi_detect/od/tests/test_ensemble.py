import pytest
import torch

from alibi_detect.od.pytorch import ensemble
from alibi_detect.exceptions import NotFittedError


def test_pval_normalizer():
    """Test the PValNormalizer

    - Test the PValNormalizer correctly normalizes data passed to it
    - Test the PValNormalizer throws the correct errors if not fit
    """
    normalizer = ensemble.PValNormalizer()
    x = torch.randn(3, 10)
    x_ref = torch.randn(64, 10)
    # unfit normalizer raises exception
    with pytest.raises(NotFittedError) as err:
        normalizer(x)
    assert err.value.args[0] == 'PValNormalizer has not been fit!'

    normalizer.fit(x_ref)
    x_norm = normalizer(x)

    # compute the p-values explicitly and compare to the normalizer
    # output.
    assert torch.all(0 < x_norm)
    assert torch.all(x_norm < 1)
    for i in range(3):
        for j in range(10):
            comp_pval = ((x_ref[:, j] > x[i][j]).to(torch.float32)).sum() + 1
            comp_pval /= (x_ref.shape[0] + 1)
            normalizer_pval = x_norm[i][j].to(torch.float32)
            assert torch.isclose(1 - comp_pval, normalizer_pval, atol=1e-4)

    # Test the scriptability of the normalizer
    normalizer = torch.jit.script(normalizer)
    x_norm_2 = normalizer(x)
    assert torch.all(x_norm_2 == x_norm)


def test_shift_and_scale_normalizer():
    """Test the ShiftAndScaleNormalizer

    - Test the ShiftAndScaleNormalizer correctly normalizes data passed to it
    - Test the ShiftAndScaleNormalizer throws the correct errors if not fit.
    """
    normalizer = ensemble.ShiftAndScaleNormalizer()
    x = torch.randn(3, 10) * 3 + 2
    x_ref = torch.randn(5000, 10) * 3 + 2

    # unfit normalizer raises exception
    with pytest.raises(NotFittedError) as err:
        normalizer(x)
    assert err.value.args[0] == 'ShiftAndScaleNormalizer has not been fit!'

    # test the normalizer correctly shifts and scales the data
    normalizer.fit(x_ref)
    x_norm = normalizer(x)
    assert torch.isclose(x_norm.mean(), torch.tensor(0.), atol=0.1)
    assert torch.isclose(x_norm.std(), torch.tensor(1.), atol=0.1)

    # Test the scriptability of the normalizer
    normalizer = torch.jit.script(normalizer)
    x_norm_2 = normalizer(x)
    assert torch.all(x_norm_2 == x_norm)


def test_average_aggregator():
    """Test the AverageAggregator

    - Test the AverageAggregator correctly aggregates data passed to it.
    - Test the AverageAggregator can be torch scripted
    """
    aggregator = ensemble.AverageAggregator()
    scores = torch.randn((3, 10))

    # test the aggregator correctly averages the scores
    aggregated_scores = aggregator(scores)
    assert torch.all(torch.isclose(aggregated_scores, scores.mean(dim=1)))
    assert aggregated_scores.shape == (3, )

    # test the scriptability of the aggregator
    aggregator = torch.jit.script(aggregator)
    aggregated_scores_2 = aggregator(scores)
    assert torch.all(aggregated_scores_2 == aggregated_scores)


def test_weighted_average_aggregator():
    """Test the AverageAggregator

    - Test the AverageAggregator correctly aggregates data passed to it
    - Test the AverageAggregator throws an error if the weights are not valid
    - Test the AverageAggregator can be torch scripted
    """
    weights = abs(torch.randn((10)))

    with pytest.raises(ValueError) as err:
        aggregator = ensemble.AverageAggregator(weights=weights)
    assert err.value.args[0] == 'Weights must sum to 1.'

    # test the aggregator correctly weights the scores when computing the
    # average
    weights /= weights.sum()
    aggregator = ensemble.AverageAggregator(weights=weights)
    scores = torch.randn((3, 10))
    aggregated_scores = aggregator(scores)
    torch.allclose(aggregated_scores, (weights @ scores.T))
    assert aggregated_scores.shape == (3, )

    # test the scriptability of the aggregator
    aggregator = torch.jit.script(aggregator)
    aggregated_scores_2 = aggregator(scores)
    assert torch.all(aggregated_scores_2 == aggregated_scores)


def test_topk_aggregator():
    """Test the TopKAggregator

    - Test the TopKAggregator correctly aggregates data passed to it
    - Test the TopKAggregator can be torch scripted
    """
    aggregator = ensemble.TopKAggregator(k=4)
    scores = torch.randn((3, 10))

    # test the aggregator correctly computes the top k scores
    aggregated_scores = aggregator(scores)
    assert aggregated_scores.shape == (3, )
    scores_sorted, _ = torch.sort(scores)
    torch.allclose(scores_sorted[:, -4:].mean(dim=1), aggregated_scores)

    # test the scriptability of the aggregator
    aggregator = torch.jit.script(aggregator)
    aggregated_scores_2 = aggregator(scores)
    assert torch.all(aggregated_scores_2 == aggregated_scores)


def test_max_aggregator():
    """Test the MaxAggregator

    - Test the MaxAggregator correctly aggregates data passed to it
    - Test the MaxAggregator can be torch scripted
    """
    aggregator = ensemble.MaxAggregator()
    scores = torch.randn((3, 10))

    # test the aggregator correctly computes the max scores
    aggregated_scores = aggregator(scores)
    assert aggregated_scores.shape == (3, )
    max_vals, _ = scores.max(dim=1)
    torch.all(max_vals == aggregated_scores)

    # test the scriptability of the aggregator
    aggregator = torch.jit.script(aggregator)
    aggregated_scores_2 = aggregator(scores)
    assert torch.all(aggregated_scores_2 == aggregated_scores)


def test_min_aggregator():
    """Test the MinAggregator

    - Test the MinAggregator correctly aggregates data passed to it
    - Test the MinAggregator can be torch scripted
    """
    aggregator = ensemble.MinAggregator()
    scores = torch.randn((3, 10))

    # test the aggregator correctly computes the min scores
    aggregated_scores = aggregator(scores)
    assert aggregated_scores.shape == (3, )
    min_vals, _ = scores.min(dim=1)
    torch.all(min_vals == aggregated_scores)

    # test the scriptability of the aggregator
    aggregator = torch.jit.script(aggregator)
    aggregated_scores_2 = aggregator(scores)
    assert torch.all(aggregated_scores_2 == aggregated_scores)


@pytest.mark.parametrize('aggregator', ['AverageAggregator', 'MaxAggregator', 'MinAggregator', 'TopKAggregator'])
@pytest.mark.parametrize('normalizer', ['PValNormalizer', 'ShiftAndScaleNormalizer'])
def test_ensembler(aggregator, normalizer):
    """Test the Ensembler for each combination of aggregator and normalizer

    - Test the ensembler correctly aggregates and normalizes the scores
    - Test the ensembler can be torch scripted
    """
    aggregator = getattr(ensemble, aggregator)()
    normalizer = getattr(ensemble, normalizer)()
    ensembler = ensemble.Ensembler(aggregator=aggregator, normalizer=normalizer)

    x = torch.randn(3, 10)
    x_ref = torch.randn(64, 10)

    # test the ensembler correctly aggregates and normalizes the scores
    ensembler.fit(x_ref)
    x_norm = ensembler(x)

    # test the scriptability of the ensembler
    ensembler = torch.jit.script(ensembler)
    x_norm_2 = ensembler(x)
    assert torch.all(x_norm_2 == x_norm)
