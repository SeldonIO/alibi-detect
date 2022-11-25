import pytest
import numpy as np
import torch

from alibi_detect.od.knn import KNN
from alibi_detect.od.backend import AverageAggregatorTorch, TopKAggregatorTorch, MaxAggregatorTorch, \
    MinAggregatorTorch, ShiftAndScaleNormaliserTorch, PValNormaliserTorch


def make_knn_detector(k=5, aggregator=None, normaliser=None):
    knn_detector = KNN(k=k, aggregator=aggregator, normaliser=normaliser)
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
    assert y['scores'][0] > 5
    assert y['scores'][1] < 1
    assert y['threshold_inferred']
    assert y['threshold'] is not None
    assert y['p_vals'].all()
    assert (y['preds'] == [True, False]).all()


@pytest.mark.parametrize("aggregator", [AverageAggregatorTorch, lambda: TopKAggregatorTorch(k=7),
                                        MaxAggregatorTorch, MinAggregatorTorch])
@pytest.mark.parametrize("normaliser", [ShiftAndScaleNormaliserTorch, PValNormaliserTorch, lambda: None])
def test_unfitted_knn_ensemble(aggregator, normaliser):
    knn_detector = KNN(
        k=[8, 9, 10],
        aggregator=aggregator(),
        normaliser=normaliser()
    )
    x = np.array([[0, 10], [0.1, 0]])
    with pytest.raises(ValueError) as err:
        _ = knn_detector.predict(x)
    assert str(err.value) == 'KNNTorch has not been fit!'


@pytest.mark.parametrize("aggregator", [AverageAggregatorTorch, lambda: TopKAggregatorTorch(k=7),
                                        MaxAggregatorTorch, MinAggregatorTorch])
@pytest.mark.parametrize("normaliser", [ShiftAndScaleNormaliserTorch, PValNormaliserTorch, lambda: None])
def test_fitted_knn_ensemble(aggregator, normaliser):
    knn_detector = KNN(
        k=[8, 9, 10],
        aggregator=aggregator(),
        normaliser=normaliser()
    )
    x_ref = np.random.randn(100, 2)
    knn_detector.fit(x_ref)
    x = np.array([[0, 10], [0, 0.1]])
    y = knn_detector.predict(x)
    assert y['scores'].all()
    assert not y['threshold_inferred']
    assert y['threshold'] is None
    assert y['preds'] is None
    assert y['p_vals'] is None


@pytest.mark.parametrize("aggregator", [AverageAggregatorTorch, lambda: TopKAggregatorTorch(k=7),
                                        MaxAggregatorTorch, MinAggregatorTorch])
@pytest.mark.parametrize("normaliser", [ShiftAndScaleNormaliserTorch, PValNormaliserTorch, lambda: None])
def test_fitted_knn_ensemble_predict(aggregator, normaliser):
    knn_detector = make_knn_detector(
        k=[8, 9, 10],
        aggregator=aggregator(),
        normaliser=normaliser()
    )
    x = np.array([[0, 10], [0, 0.1]])
    y = knn_detector.predict(x)
    assert y['threshold_inferred']
    assert y['threshold'] is not None
    assert y['p_vals'].all()
    assert (y['preds'] == [True, False]).all()


def test_incorrect_knn_ensemble_init():
    with pytest.raises(ValueError) as err:
        KNN(k=[8, 9, 10])
    assert str(err.value) == ("k=[8, 9, 10] is type <class 'list'> but aggregator is None, you must specify at least an"
                              " aggregator if you want to use the knn detector ensemble like this.")


@pytest.mark.parametrize("aggregator", [AverageAggregatorTorch, lambda: TopKAggregatorTorch(k=7),
                                        MaxAggregatorTorch, MinAggregatorTorch])
@pytest.mark.parametrize("normaliser", [ShiftAndScaleNormaliserTorch, PValNormaliserTorch, lambda: None])
def test_knn_ensemble_torch_script(aggregator, normaliser):
    knn_detector = make_knn_detector(k=[5, 6, 7], aggregator=aggregator(), normaliser=normaliser())
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
