import imp
import numpy as np
import os

from alibi_detect.od.deepsvdd import DeepSVDD
from alibi_detect.od.backends import DeepSVDDTorch
from alibi_detect.od.loading import load_detector
from alibi_detect.od.config import write_config, ModelWrapper
from torch.utils.data import DataLoader
from alibi_detect.utils.pytorch.data import TorchDataset

import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.fc1(x)


def test_deepsvdd_config(tmp_path):
    tmp_path = './example-test'
    model = Model()
    deepsvdd_detector = DeepSVDD(model)

    path = deepsvdd_detector.save(tmp_path)
    loaded_deepsvdd_detector = load_detector(path)

    assert isinstance(loaded_deepsvdd_detector, DeepSVDD)
    assert loaded_deepsvdd_detector.backend.__class__.__name__ == DeepSVDDTorch.__name__
    assert loaded_deepsvdd_detector.backend.dataloader.func == DataLoader
    assert loaded_deepsvdd_detector.backend.dataset == TorchDataset
    assert isinstance(loaded_deepsvdd_detector.model, Model)
    assert isinstance(loaded_deepsvdd_detector.original_model, ModelWrapper)
    
