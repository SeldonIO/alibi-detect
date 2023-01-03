import pytest
import numpy as np
import torch

from alibi_detect.od.knn import KNN
from alibi_detect.od.backend import AverageAggregatorTorch, TopKAggregatorTorch, MaxAggregatorTorch, \
    MinAggregatorTorch, ShiftAndScaleNormalizerTorch, PValNormalizerTorch
from sklearn.datasets import make_moons


def make_knn_detector(k=5, aggregator=None, normalizer=None):
    knn_detector = KNN(k=k, aggregator=aggregator, normalizer=normalizer)
    x_ref = np.random.randn(100, 2)
    knn_detector.fit(x_ref)
    knn_detector.infer_threshold(x_ref, 0.1)
    return knn_detector


def test_unfitted_knn_single_score():
    knn_detector = KNN(k=10)
    x = np.array([[0, 10], [0.1, 0]])
    with pytest.raises(ValueError) as err:
        _ = knn_detector.predict(x)
    assert str(err.value) == 'KNNTorch has not been fit!'


def test_fitted_knn_single_score():
    knn_detector = KNN(k=10)
    x_ref = np.random.randn(100, 2)
    knn_detector.fit(x_ref)
    x = np.array([[0, 10], [0.1, 0]])
    y = knn_detector.predict(x)
    y = y['data']
    assert y['scores'][0] > 5
    assert y['scores'][1] < 1

    assert not y['threshold_inferred']
    assert y['threshold'] is None
    assert y['preds'] is None
    assert y['p_vals'] is None


def test_default_knn_ensemble_init():
    knn_detector = KNN(k=[8, 9, 10])
    x_ref = np.random.randn(100, 2)
    knn_detector.fit(x_ref)
    x = np.array([[0, 10], [0.1, 0]])
    y = knn_detector.predict(x)
    y = y['data']
    assert y['scores'][0] > 5
    assert y['scores'][1] < 1

    assert not y['threshold_inferred']
    assert y['threshold'] is None
    assert y['preds'] is None
    assert y['p_vals'] is None


def test_fitted_knn_predict():
    knn_detector = make_knn_detector(k=10)
    x_ref = np.random.randn(100, 2)
    knn_detector.infer_threshold(x_ref, 0.1)
    x = np.array([[0, 10], [0, 0.1]])
    y = knn_detector.predict(x)
    y = y['data']
    assert y['scores'][0] > 5
    assert y['scores'][1] < 1
    assert y['threshold_inferred']
    assert y['threshold'] is not None
    assert y['p_vals'].all()
    assert (y['preds'] == [True, False]).all()


@pytest.mark.parametrize("aggregator", [AverageAggregatorTorch, lambda: TopKAggregatorTorch(k=7),
                                        MaxAggregatorTorch, MinAggregatorTorch])
@pytest.mark.parametrize("normalizer", [ShiftAndScaleNormalizerTorch, PValNormalizerTorch, lambda: None])
def test_unfitted_knn_ensemble(aggregator, normalizer):
    knn_detector = KNN(
        k=[8, 9, 10],
        aggregator=aggregator(),
        normalizer=normalizer()
    )
    x = np.array([[0, 10], [0.1, 0]])
    with pytest.raises(ValueError) as err:
        _ = knn_detector.predict(x)
    assert str(err.value) == 'KNNTorch has not been fit!'


@pytest.mark.parametrize("aggregator", [AverageAggregatorTorch, lambda: TopKAggregatorTorch(k=7),
                                        MaxAggregatorTorch, MinAggregatorTorch])
@pytest.mark.parametrize("normalizer", [ShiftAndScaleNormalizerTorch, PValNormalizerTorch, lambda: None])
def test_fitted_knn_ensemble(aggregator, normalizer):
    knn_detector = KNN(
        k=[8, 9, 10],
        aggregator=aggregator(),
        normalizer=normalizer()
    )
    x_ref = np.random.randn(100, 2)
    knn_detector.fit(x_ref)
    x = np.array([[0, 10], [0, 0.1]])
    y = knn_detector.predict(x)
    y = y['data']
    assert y['scores'].all()
    assert not y['threshold_inferred']
    assert y['threshold'] is None
    assert y['preds'] is None
    assert y['p_vals'] is None


@pytest.mark.parametrize("aggregator", [AverageAggregatorTorch, lambda: TopKAggregatorTorch(k=7),
                                        MaxAggregatorTorch, MinAggregatorTorch])
@pytest.mark.parametrize("normalizer", [ShiftAndScaleNormalizerTorch, PValNormalizerTorch, lambda: None])
def test_fitted_knn_ensemble_predict(aggregator, normalizer):
    knn_detector = make_knn_detector(
        k=[8, 9, 10],
        aggregator=aggregator(),
        normalizer=normalizer()
    )
    x = np.array([[0, 10], [0, 0.1]])
    y = knn_detector.predict(x)
    y = y['data']
    assert y['threshold_inferred']
    assert y['threshold'] is not None
    assert y['p_vals'].all()
    assert (y['preds'] == [True, False]).all()
    

@pytest.mark.parametrize("aggregator", [AverageAggregatorTorch, lambda: TopKAggregatorTorch(k=7),
                                        MaxAggregatorTorch, MinAggregatorTorch])
@pytest.mark.parametrize("normalizer", [ShiftAndScaleNormalizerTorch, PValNormalizerTorch, lambda: None])
def test_knn_ensemble_torch_script(aggregator, normalizer):
    knn_detector = make_knn_detector(k=[5, 6, 7], aggregator=aggregator(), normalizer=normalizer())
    tsknn = torch.jit.script(knn_detector.backend)
    x = torch.tensor([[0, 10], [0, 0.1]])
    y = tsknn(x)
    assert torch.all(y == torch.tensor([True, False]))


def test_knn_single_torchscript():
    knn_detector = make_knn_detector(k=5)
    tsknn = torch.jit.script(knn_detector.backend)
    x = torch.tensor([[0, 10], [0, 0.1]])
    y = tsknn(x)
    assert torch.all(y == torch.tensor([True, False]))


@pytest.mark.parametrize("aggregator", [AverageAggregatorTorch, lambda: TopKAggregatorTorch(k=7),
                                        MaxAggregatorTorch, MinAggregatorTorch])
@pytest.mark.parametrize("normalizer", [ShiftAndScaleNormalizerTorch, PValNormalizerTorch, lambda: None])
def test_knn_ensemble_integration(aggregator, normalizer):
    knn_detector = KNN(
        k=[10, 14, 18],
        aggregator=aggregator(),
        normalizer=normalizer()
    )
    X_ref, _ = make_moons(1001, shuffle=True, noise=0.05, random_state=None)
    X_ref, x_inlier = X_ref[0:1000], X_ref[1000][None]
    knn_detector.fit(X_ref)
    knn_detector.infer_threshold(X_ref, 0.1)
    result = knn_detector.predict(x_inlier)
    result = result['data']['preds'][0]
    assert not result

    x_outlier = np.array([[-1, 1.5]])
    result = knn_detector.predict(x_outlier)
    result = result['data']['preds'][0]
    assert result

    tsknn = torch.jit.script(knn_detector.backend)
    x = torch.tensor([x_inlier[0], x_outlier[0]], dtype=torch.float32)
    y = tsknn(x)
    assert torch.all(y == torch.tensor([False, True]))


def test_knn_integration():
    knn_detector = KNN(k=18)
    X_ref, _ = make_moons(1001, shuffle=True, noise=0.05, random_state=None)
    X_ref, x_inlier = X_ref[0:1000], X_ref[1000][None]
    knn_detector.fit(X_ref)
    knn_detector.infer_threshold(X_ref, 0.1)
    result = knn_detector.predict(x_inlier)
    result = result['data']['preds'][0]
    assert not result

    x_outlier = np.array([[-1, 1.5]])
    result = knn_detector.predict(x_outlier)
    result = result['data']['preds'][0]
    assert result

    tsknn = torch.jit.script(knn_detector.backend)
    x = torch.tensor([x_inlier[0], x_outlier[0]], dtype=torch.float32)
    y = tsknn(x)
    assert torch.all(y == torch.tensor([False, True]))
