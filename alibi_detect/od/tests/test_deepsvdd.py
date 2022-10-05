import numpy as np
import os

from alibi_detect.od.deepsvdd import DeepSVDD
# from alibi_detect.od.loading import load_detector
# from alibi_detect.od.config import write_config
# from alibi_detect.od.aggregation import AverageAggregator, ShiftAndScaleNormaliser, PValNormaliser, TopKAggregator
# from alibi_detect.od.backends import KNNTorch
# from alibi_detect.od.backends import GaussianRBF

import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.fc1(x)


def test_deepsvdd_config(tmp_path):
    model = Model()
    deepsvdd_detector = DeepSVDD(model)
    # print(deepsvdd_detector.config['model'])
    config = deepsvdd_detector.serialize(tmp_path)
    # print(deepsvdd_detector.model)
    # print(deepsvdd_detector.original_model)
    # print(deepsvdd_detector.config['model'])
    print(deepsvdd_detector.get_config())

    # write_config(knn_detector.serialize(tmp_path), tmp_path)
    # loaded_knn_detector = load_detector(os.path.join(tmp_path, 'config.toml'))

    # assert isinstance(loaded_knn_detector, KNN)
    # assert isinstance(loaded_knn_detector.aggregator, TopKAggregator)
    # assert isinstance(loaded_knn_detector.normaliser, ShiftAndScaleNormaliser)
    # assert isinstance(loaded_knn_detector.kernel, GaussianRBF)
    # assert loaded_knn_detector.k == [8, 9, 10]
    # assert loaded_knn_detector.kernel.config['sigma'] == [1.0]
    # assert loaded_knn_detector.aggregator.k == 5
    # assert loaded_knn_detector.backend.__name__ == KNNTorch.__name__
    # assert loaded_knn_detector.kernel.init_sigma_fn() == 'test'

    assert 1 == 0
