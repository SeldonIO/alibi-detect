import pytest
import numpy as np

from alibi_detect.od.knn import KNN
from alibi_detect.od.backend import AverageAggregatorTorch, TopKAggregatorTorch, MaxAggregatorTorch, \
    MinAggregatorTorch, ShiftAndScaleNormaliserTorch, PValNormaliserTorch


def test_knn_single():
    knn_detector = KNN(k=10)
    x_ref = np.random.randn(100, 2)
    knn_detector.fit(x_ref)
    x = np.array([[0, 10]])
    assert knn_detector.predict(x)['scores'] > 5

    x = np.array([[0, 0.1]])
    assert knn_detector.predict(x)['scores'] < 1

    knn_detector.infer_threshold(x_ref, 0.1)

    x = np.array([[0, 10]])
    pred = knn_detector.predict(x)
    assert pred['scores'] > 5
    assert pred['preds']
    assert pred['p_vals'] < 0.05

    x = np.array([[0, 0.1]])
    pred = knn_detector.predict(x)
    assert pred['scores'] < 1
    assert not pred['preds']
    assert pred['p_vals'] > 0.7


@pytest.mark.parametrize("aggregator", [AverageAggregatorTorch, lambda: TopKAggregatorTorch(k=7),
                                        MaxAggregatorTorch, MinAggregatorTorch])
@pytest.mark.parametrize("normaliser", [ShiftAndScaleNormaliserTorch, PValNormaliserTorch, lambda: None])
def test_knn_ensemble(aggregator, normaliser):
    knn_detector = KNN(
        k=[8, 9, 10],
        aggregator=aggregator(),
        normaliser=normaliser()
    )
    x_ref = np.random.randn(100, 2)
    knn_detector.fit(x_ref)
    x = np.array([[0, 10], [0, 0.1]])
    knn_detector.infer_threshold(x_ref, 0.1)
    pred = knn_detector.predict(x)

    assert np.all(pred['preds'] == [True, False])


def test_incorrect_knn_ensemble_init():
    with pytest.raises(ValueError) as err:
        KNN(k=[8, 9, 10])
    assert str(err.value) == ("k=[8, 9, 10] is type <class 'list'> but aggregator is None, you must specify at least an"
                              " aggregator if you want to use the knn detector ensemble like this.")
