import pytest
import numpy as np
import torch

from alibi_detect.od._lof import LOF
from alibi_detect.od import AverageAggregator, TopKAggregator, MaxAggregator, \
    MinAggregator, ShiftAndScaleNormalizer, PValNormalizer
from alibi_detect.base import NotFitException

from sklearn.datasets import make_moons


def make_lof_detector(k=5, aggregator=None, normalizer=None):
    lof_detector = LOF(
        k=k, aggregator=aggregator,
        normalizer=normalizer
    )
    x_ref = np.random.randn(100, 2)
    lof_detector.fit(x_ref)
    lof_detector.infer_threshold(x_ref, 0.1)
    return lof_detector


def test_unfitted_lof_single_score():
    lof_detector = LOF(k=10)
    x = np.array([[0, 10], [0.1, 0]])

    # test predict raises exception when not fitted
    with pytest.raises(NotFitException) as err:
        _ = lof_detector.predict(x)
    assert str(err.value) == 'LOFTorch has not been fit!'


@pytest.mark.parametrize('k', [10, [8, 9, 10]])
def test_fitted_lof_single_score(k):
    lof_detector = LOF(k=k)
    x_ref = np.random.randn(100, 2)
    lof_detector.fit(x_ref)
    x = np.array([[0, 10], [0.1, 0]])
    # test fitted but not threshold inferred detectors
    # can still score data using the predict method.
    y = lof_detector.predict(x)
    y = y['data']
    assert y['instance_score'][0] > 7
    assert y['instance_score'][1] < 2

    assert not y['threshold_inferred']
    assert y['threshold'] is None
    assert y['is_outlier'] is None
    assert y['p_value'] is None


def test_incorrect_lof_ensemble_init():
    # test lof ensemble with aggregator passed as None raises exception

    with pytest.raises(ValueError) as err:
        LOF(k=[8, 9, 10], aggregator=None)
    assert str(err.value) == ('If `k` is a `np.ndarray`, `list` or `tuple`, '
                              'the `aggregator` argument cannot be ``None``.')


def test_fitted_lof_predict():
    """
    Test that a detector fitted on data and with threshold inferred correctly, will score
    and label outliers, as well as return the p-values using the predict method. Also Check
    that the score method gives the same results.
    """

    lof_detector = make_lof_detector(k=10)
    x_ref = np.random.randn(100, 2)
    lof_detector.infer_threshold(x_ref, 0.1)
    x = np.array([[0, 10], [0, 0.1]])

    y = lof_detector.predict(x)
    y = y['data']
    scores = lof_detector.score(x)
    assert np.all(y['instance_score'] == scores)
    assert y['instance_score'][0] > 7
    assert y['instance_score'][1] < 2
    assert y['threshold_inferred']
    assert y['threshold'] is not None
    assert y['p_value'].all()
    assert (y['is_outlier'] == [True, False]).all()


@pytest.mark.parametrize("aggregator", [AverageAggregator, lambda: TopKAggregator(k=7),
                                        MaxAggregator, MinAggregator])
@pytest.mark.parametrize("normalizer", [ShiftAndScaleNormalizer, PValNormalizer, lambda: None])
def test_unfitted_lof_ensemble(aggregator, normalizer):
    lof_detector = LOF(
        k=[8, 9, 10],
        aggregator=aggregator(),
        normalizer=normalizer()
    )
    x = np.array([[0, 10], [0.1, 0]])

    # Test unfit lof ensemble raises exception when calling predict method.
    with pytest.raises(NotFitException) as err:
        _ = lof_detector.predict(x)
    assert str(err.value) == 'LOFTorch has not been fit!'


@pytest.mark.parametrize("aggregator", [AverageAggregator, lambda: TopKAggregator(k=7),
                                        MaxAggregator, MinAggregator])
@pytest.mark.parametrize("normalizer", [ShiftAndScaleNormalizer, PValNormalizer, lambda: None])
def test_fitted_lof_ensemble(aggregator, normalizer):
    lof_detector = LOF(
        k=[8, 9, 10],
        aggregator=aggregator(),
        normalizer=normalizer()
    )
    x_ref = np.random.randn(100, 2)
    lof_detector.fit(x_ref)
    x = np.array([[0, 10], [0, 0.1]])

    # test fitted but not threshold inferred detectors can still score data using the predict method.
    y = lof_detector.predict(x)
    y = y['data']
    assert y['instance_score'].all()
    assert not y['threshold_inferred']
    assert y['threshold'] is None
    assert y['is_outlier'] is None
    assert y['p_value'] is None


@pytest.mark.parametrize("aggregator", [AverageAggregator, lambda: TopKAggregator(k=7),
                                        MaxAggregator, MinAggregator])
@pytest.mark.parametrize("normalizer", [ShiftAndScaleNormalizer, PValNormalizer, lambda: None])
def test_fitted_lof_ensemble_predict(aggregator, normalizer):
    lof_detector = make_lof_detector(
        k=[8, 9, 10],
        aggregator=aggregator(),
        normalizer=normalizer()
    )
    x = np.array([[0, 10], [0, 0.1]])

    # test fitted detectors with inferred thresholds can score data using the predict method.
    y = lof_detector.predict(x)
    y = y['data']
    assert y['threshold_inferred']
    assert y['threshold'] is not None
    assert y['p_value'].all()
    assert (y['is_outlier'] == [True, False]).all()


@pytest.mark.parametrize("aggregator", [AverageAggregator, lambda: TopKAggregator(k=7),
                                        MaxAggregator, MinAggregator])
@pytest.mark.parametrize("normalizer", [ShiftAndScaleNormalizer, PValNormalizer, lambda: None])
def test_lof_ensemble_torch_script(aggregator, normalizer):
    lof_detector = make_lof_detector(k=[5, 6, 7], aggregator=aggregator(), normalizer=normalizer())
    tslof = torch.jit.script(lof_detector.backend)
    x = torch.tensor([[0, 10], [0, 0.1]])

    # test torchscripted ensemble lof detector can be saved and loaded correctly.
    y = tslof(x)
    assert torch.all(y == torch.tensor([True, False]))


def test_lof_single_torchscript():
    lof_detector = make_lof_detector(k=5)
    tslof = torch.jit.script(lof_detector.backend)
    x = torch.tensor([[0, 10], [0, 0.1]])

    # test torchscripted single lof detector can be saved and loaded correctly.
    y = tslof(x)
    assert torch.all(y == torch.tensor([True, False]))


@pytest.mark.parametrize("aggregator", [AverageAggregator, lambda: TopKAggregator(k=7),
                                        MaxAggregator, MinAggregator, lambda: 'AverageAggregator',
                                        lambda: 'TopKAggregator', lambda: 'MaxAggregator',
                                        lambda: 'MinAggregator'])
@pytest.mark.parametrize("normalizer", [ShiftAndScaleNormalizer, PValNormalizer, lambda: None,
                                        lambda: 'ShiftAndScaleNormalizer', lambda: 'PValNormalizer'])
def test_lof_ensemble_integration(aggregator, normalizer):
    """Test lof ensemble detector on moons dataset.

    Tests ensemble lof detector with every combination of aggregator and normalizer on the moons dataset.
    Fits and infers thresholds in each case. Verifies that the detector can correctly detect inliers
    and outliers and that it can be serialized using the torchscript.
    """

    lof_detector = LOF(
        k=[10, 14, 18],
        aggregator=aggregator(),
        normalizer=normalizer()
    )
    X_ref, _ = make_moons(1001, shuffle=True, noise=0.05, random_state=None)
    X_ref, x_inlier = X_ref[0:1000], X_ref[1000][None]
    lof_detector.fit(X_ref)
    lof_detector.infer_threshold(X_ref, 0.1)
    result = lof_detector.predict(x_inlier)
    result = result['data']['is_outlier'][0]
    assert not result

    x_outlier = np.array([[-1, 1.5]])
    result = lof_detector.predict(x_outlier)
    result = result['data']['is_outlier'][0]
    assert result

    tslof = torch.jit.script(lof_detector.backend)
    x = torch.tensor([x_inlier[0], x_outlier[0]], dtype=torch.float32)
    y = tslof(x)
    assert torch.all(y == torch.tensor([False, True]))


def test_lof_integration():
    """Test lof detector on moons dataset.

    Tests lof detector on the moons dataset. Fits and infers thresholds and verifies that the detector can
    correctly detect inliers and outliers. Checks that it can be serialized using the torchscript.
    """
    lof_detector = LOF(k=18)
    X_ref, _ = make_moons(1001, shuffle=True, noise=0.05, random_state=None)
    X_ref, x_inlier = X_ref[0:1000], X_ref[1000][None]
    lof_detector.fit(X_ref)
    lof_detector.infer_threshold(X_ref, 0.1)
    result = lof_detector.predict(x_inlier)
    result = result['data']['is_outlier'][0]
    assert not result

    x_outlier = np.array([[-1, 1.5]])
    result = lof_detector.predict(x_outlier)
    result = result['data']['is_outlier'][0]
    assert result

    tslof = torch.jit.script(lof_detector.backend)
    x = torch.tensor([x_inlier[0], x_outlier[0]], dtype=torch.float32)
    y = tslof(x)
    assert torch.all(y == torch.tensor([False, True]))
