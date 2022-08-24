import numpy as np
from alibi_detect.od.ensemble import Ensemble
from alibi_detect.od.knn import KNN
from alibi_detect.od.aggregation import AverageAggregator, PValNormaliser
from alibi_detect.od.processor import ParallelProcessor


def test_ensemble():
    knn_detectors = [KNN(k=k+1) for k in range(10)]
    ensemble_detector = Ensemble(
        detectors=knn_detectors, 
        aggregator=AverageAggregator(), 
        normaliser=PValNormaliser()
    )
    x_ref = np.random.randn(100, 2)
    ensemble_detector.fit(x_ref)
    x = np.array([[0, 0.1]])
    ensemble_detector.infer_threshold(x_ref, 0.1)
    assert ensemble_detector.predict(x)['p_vals'].item() > 0.5


def test_ensemble_parrallel():
    knn_detectors = [KNN(k=k+1) for k in range(10)]
    ensemble_detector = Ensemble(
        detectors=knn_detectors, 
        aggregator=AverageAggregator(), 
        normaliser=PValNormaliser(),
        processor=ParallelProcessor
    )
    x_ref = np.random.randn(100, 2)
    ensemble_detector.fit(x_ref)
    x = np.array([[0, 0.1]])
    ensemble_detector.infer_threshold(x_ref, 0.1)
    assert ensemble_detector.predict(x)['p_vals'].item() > 0.5