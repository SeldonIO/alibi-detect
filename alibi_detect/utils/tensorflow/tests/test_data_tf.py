from itertools import product
import numpy as np
import pytest
from alibi_detect.utils.tensorflow.data import TFDataset

# test on numpy array and list
n, f = 100, 5
shape = (n, f)
xtype = [list, np.ndarray]
shuffle = [True, False]
batch_size = [2, 10]
tests_ds = list(product(xtype, batch_size, shuffle))
n_tests_ds = len(tests_ds)


@pytest.fixture
def ds_params(request):
    return tests_ds[request.param]


@pytest.mark.parametrize('ds_params', list(range(n_tests_ds)), indirect=True)
def test_torchdataset(ds_params):
    xtype, batch_size, shuffle = ds_params
    x = np.random.randn(*shape)
    y = np.random.randn(*(n,))
    if xtype == list:
        x = list(x)
    ds = TFDataset(x, y, batch_size=batch_size, shuffle=shuffle)
    for step, data in enumerate(ds):
        pass
    if xtype == list:
        assert len(data[0]) == batch_size and data[0][0].shape == (f,)
    else:
        assert data[0].shape == (batch_size, f)
    assert data[1].shape == (batch_size,)
    assert step == len(ds) - 1
    if not shuffle:
        assert (data[0][-1] == x[-1 - (n % batch_size)]).all()
