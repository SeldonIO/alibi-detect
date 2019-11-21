import numpy as np
import pandas as pd
from alibi_detect.datasets import fetch_nab, get_list_nab
from alibi_detect.utils.data import Bunch

files = get_list_nab()


def test_list_nab():
    assert len(files) == 58


def test_fetch_nab():
    idx = np.random.choice(len(files))
    data = fetch_nab(files[idx])
    assert isinstance(data, Bunch)
    assert isinstance(data.data, pd.DataFrame) and isinstance(data.target, pd.DataFrame)
    data = fetch_nab(files[idx], return_X_y=True)
    assert isinstance(data, tuple)
    assert isinstance(data[0], pd.DataFrame) and isinstance(data[1], pd.DataFrame)
