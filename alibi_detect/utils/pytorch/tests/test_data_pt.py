import numpy as np
import pytest
from alibi_detect.utils.pytorch.data import TorchDataset

# test on numpy array and list
n, f = 100, 5
shape = (n, f)
tests_ds = [list, np.ndarray]
n_tests_ds = len(tests_ds)


@pytest.fixture
def ds_params(request):
    return tests_ds[request.param]


@pytest.mark.parametrize('ds_params', list(range(n_tests_ds)), indirect=True)
def test_torchdataset(ds_params):
    xtype = ds_params
    x = np.random.randn(*shape)
    y = np.random.randn(*(n,))
    if xtype == list:
        x = list(x)
    ds = TorchDataset(x, y)
    for step, data in enumerate(ds):
        pass
    assert data[0].shape == (f,) and data[1].shape == ()
    assert step == len(ds) - 1
