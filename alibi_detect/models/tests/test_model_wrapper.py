from alibi_detect.models.model_wrapper import ModelWrapper
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.fc1(x)


def test_model_wrapper():
    mw = ModelWrapper(Model())
    print(mw.forward)
    assert 1 == 0