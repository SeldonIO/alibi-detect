import pytest
import numpy as np

from alibi_detect.od.knn import KNN
from alibi_detect.od.backend.torch.ensemble import AverageAggregator, TopKAggregator, MaxAggregator, \
    MinAggregator, ShiftAndScaleNormaliser, PValNormaliser


def test_knn_single():
    knn_detector = KNN(k=10)
    x_ref = np.random.randn(100, 2)
    knn_detector.fit(x_ref)
    x = np.array([[0, 10]])
    assert knn_detector.predict(x)['raw_scores'] > 5

    x = np.array([[0, 0.1]])
    assert knn_detector.predict(x)['raw_scores'] < 1

    knn_detector.infer_threshold(x_ref, 0.1)

    x = np.array([[0, 10]])
    pred = knn_detector.predict(x)
    assert pred['raw_scores'] > 5
    assert pred['preds']
    assert pred['p_vals'] < 0.05

    x = np.array([[0, 0.1]])
    pred = knn_detector.predict(x)
    assert pred['raw_scores'] < 1
    assert not pred['preds']
    assert pred['p_vals'] > 0.7


@pytest.mark.parametrize("aggregator", [AverageAggregator, lambda: TopKAggregator(k=7), MaxAggregator, MinAggregator])
@pytest.mark.parametrize("normaliser", [ShiftAndScaleNormaliser, PValNormaliser, lambda: None])
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
    if isinstance(knn_detector.normaliser, ShiftAndScaleNormaliser):
        assert np.all(pred['normalised_scores'][0] > 1)
        assert np.all(pred['normalised_scores'][1] < 0)
    elif isinstance(knn_detector.normaliser, PValNormaliser):
        assert np.all(pred['normalised_scores'][0] > 0.8)
        assert np.all(pred['normalised_scores'][1] < 0.3)


@pytest.mark.parametrize("aggregator", [AverageAggregator, lambda: TopKAggregator(k=7), MaxAggregator, MinAggregator])
@pytest.mark.parametrize("normaliser", [ShiftAndScaleNormaliser, PValNormaliser, lambda: None])
def test_knn_keops(aggregator, normaliser):
    knn_detector = KNN(
        k=[8, 9, 10],
        aggregator=aggregator(),
        normaliser=normaliser(),
        backend='keops'
    )

    x_ref = np.random.randn(100, 2)
    knn_detector.fit(x_ref)
    x = np.array([[0, 10], [0, 0.1]])
    knn_detector.infer_threshold(x_ref, 0.1)
    pred = knn_detector.predict(x)

    assert np.all(pred['preds'] == [True, False])
    if isinstance(knn_detector.normaliser, ShiftAndScaleNormaliser):
        assert np.all(pred['normalised_scores'][0] > 1)
        assert np.all(pred['normalised_scores'][1] < 0)
    elif isinstance(knn_detector.normaliser, PValNormaliser):
        assert np.all(pred['normalised_scores'][0] > 0.8)
        assert np.all(pred['normalised_scores'][1] < 0.3)
