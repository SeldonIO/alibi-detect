import numpy as np
import os

from alibi_detect.od.knn import KNN
from alibi_detect.od.loading import load_detector
from alibi_detect.od.config import write_config
from alibi_detect.od.aggregation import AverageAggregator, ShiftAndScaleNormaliser, PValNormaliser, TopKAggregator
from alibi_detect.od.backends import KNNTorch
from alibi_detect.od.backends import GaussianRBF

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
    x = np.array([[0, 10], [0, 0.1]])
    knn_detector.infer_threshold(x_ref, 0.1)
    pred = knn_detector.predict(x)

    assert np.all(pred['normalised_scores'][0] > 1)
    assert np.all(pred['normalised_scores'][1] < 0) # Is this correct?
    assert np.all(pred['preds'] == [True, False])

    knn_detector = KNN(
        k=[8, 9, 10], 
        aggregator=AverageAggregator(), 
        normaliser=PValNormaliser()
    )

    x_ref = np.random.randn(100, 2)
    knn_detector.fit(x_ref)
    x = np.array([[0, 10], [0, 0.1]])
    knn_detector.infer_threshold(x_ref, 0.1)
    pred = knn_detector.predict(x)
    
    assert np.all(pred['normalised_scores'][0] > 0.8)
    assert np.all(pred['normalised_scores'][1] < 0.3)
    assert np.all(pred['preds'] == [True, False])


def test_knn_keops():
    knn_detector = KNN(
        k=[8, 9, 10], 
        aggregator=AverageAggregator(), 
        normaliser=ShiftAndScaleNormaliser(),
        backend='keops'
    )

    x_ref = np.random.randn(100, 2)
    knn_detector.fit(x_ref)
    x = np.array([[0, 10], [0, 0.1]])
    knn_detector.infer_threshold(x_ref, 0.1)
    pred = knn_detector.predict(x)

    assert np.all(pred['normalised_scores'][0] > 1)
    assert np.all(pred['normalised_scores'][1] < 0) # Is this correct?
    assert np.all(pred['preds'] == [True, False])

    knn_detector = KNN(
        k=[8, 9, 10], 
        aggregator=AverageAggregator(), 
        normaliser=PValNormaliser()
    )

    x_ref = np.random.randn(100, 2)
    knn_detector.fit(x_ref)
    x = np.array([[0, 10], [0, 0.1]])
    knn_detector.infer_threshold(x_ref, 0.1)
    pred = knn_detector.predict(x)
    
    assert np.all(pred['normalised_scores'][0] > 0.8)
    assert np.all(pred['normalised_scores'][1] < 0.3)
    assert np.all(pred['preds'] == [True, False])

def test_knn_config(tmp_path):
    knn_detector = KNN(
        k=[8, 9, 10], 
        aggregator=TopKAggregator(k=5), 
        normaliser=ShiftAndScaleNormaliser(),
        backend='pytorch',
        kernel=GaussianRBF(
            np.array([1.]),
            init_sigma_fn=lambda: 'test')
    )
    path = knn_detector.save(tmp_path)
    loaded_detector = load_detector(path)

    assert isinstance(loaded_detector, KNN)
    assert isinstance(loaded_detector.aggregator, TopKAggregator)
    assert isinstance(loaded_detector.normaliser, ShiftAndScaleNormaliser)
    assert isinstance(loaded_detector.kernel, GaussianRBF)
    assert loaded_detector.k == [8, 9, 10]
    assert loaded_detector.kernel.config['sigma'] == [1.0]
    assert loaded_detector.aggregator.k == 5
    assert loaded_detector.backend.__name__ == KNNTorch.__name__
    assert loaded_detector.kernel.init_sigma_fn() == 'test'
