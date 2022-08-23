import imp
import numpy as np
from alibi_detect.od.knn import KNN
from alibi_detect.od.aggregation import AverageAggregator, ShiftAndScaleNormaliser


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
    assert pred['preds'] == True
    assert pred['p_vals'] < 0.05

    x = np.array([[0, 0.1]])
    pred = knn_detector.predict(x)
    assert pred['raw_scores'] < 1
    assert pred['preds'] == False
    assert pred['p_vals'] > 0.7


def test_knn_ensemble(): 
    knn_detector = KNN(
        k=[8, 9, 10], 
        aggregator=AverageAggregator(), 
        normaliser=ShiftAndScaleNormaliser()
    )

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
    assert pred['preds'] == True
    assert pred['p_vals'] < 0.05

    x = np.array([[0, 0.1]])
    pred = knn_detector.predict(x)
    assert pred['raw_scores'] < 1
    assert pred['preds'] == False
    assert pred['p_vals'] > 0.7
