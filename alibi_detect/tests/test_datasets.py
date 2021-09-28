import numpy as np
import pandas as pd
import pytest
from requests import RequestException
from alibi_detect.datasets import fetch_kdd, fetch_ecg, corruption_types_cifar10c, fetch_cifar10c, \
    fetch_attack, fetch_nab, get_list_nab
from alibi_detect.utils.data import Bunch

# KDD cup dataset
target_list = ['dos', 'r2l', 'u2r', 'probe']
keep_cols_list = ['srv_count', 'serror_rate', 'srv_serror_rate',
                  'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                  'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                  'dst_host_srv_count', 'dst_host_same_srv_rate',
                  'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                  'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                  'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                  'dst_host_srv_rerror_rate']


@pytest.mark.parametrize('return_X_y', [True, False])
def test_fetch_kdd(return_X_y):
    target = np.random.choice(target_list, 2, replace=False)
    keep_cols = np.random.choice(keep_cols_list, 5, replace=False)
    try:
        data = fetch_kdd(target=target, keep_cols=keep_cols, percent10=True, return_X_y=return_X_y)
    except RequestException:
        pytest.skip('KDD dataset URL down')
    if return_X_y:
        assert isinstance(data, tuple)
        assert isinstance(data[0], np.ndarray) and isinstance(data[1], np.ndarray)
    else:
        assert isinstance(data, Bunch)
        assert isinstance(data.data, np.ndarray) and isinstance(data.target, np.ndarray)
        assert list(data.feature_names) == list(keep_cols)


# ECG dataset
@pytest.mark.parametrize('return_X_y', [True, False])
def test_fetch_ecg(return_X_y):
    try:
        data = fetch_ecg(return_X_y=return_X_y)
    except RequestException:
        pytest.skip('ECG dataset URL down')
    if return_X_y:
        assert isinstance(data, tuple)
        assert isinstance(data[0][0], np.ndarray) and isinstance(data[0][1], np.ndarray) and \
               isinstance(data[1][0], np.ndarray) and isinstance(data[1][1], np.ndarray)
    else:
        assert isinstance(data, Bunch)
        assert isinstance(data.data_train, np.ndarray) and isinstance(data.data_test, np.ndarray) and \
               isinstance(data.target_train, np.ndarray) and isinstance(data.target_test, np.ndarray)


# CIFAR-10-C dataset
corruption_list = corruption_types_cifar10c()


def test_types_cifar10c():
    assert len(corruption_list) == 19


@pytest.mark.parametrize('return_X_y', [True, False])
def test_fetch_cifar10c(return_X_y):
    corruption = list(np.random.choice(corruption_list, 5, replace=False))
    try:
        data = fetch_cifar10c(corruption=corruption, severity=2, return_X_y=return_X_y)
    except RequestException:
        pytest.skip('CIFAR-10-C dataset URL down')
    if return_X_y:
        assert isinstance(data, tuple)
        assert isinstance(data[0], np.ndarray) and isinstance(data[1], np.ndarray)
    else:
        assert isinstance(data, Bunch)
        assert isinstance(data.data, np.ndarray) and isinstance(data.target, np.ndarray)


# Attack datasets
datasets = ['cifar10']
models = ['resnet56']
attacks = ['cw', 'slide']


@pytest.mark.parametrize('return_X_y', [True, False])
def test_fetch_attack(return_X_y):
    dataset = list(np.random.choice(datasets, 1))[0]
    model = list(np.random.choice(models, 1))[0]
    attack = list(np.random.choice(attacks, 1))[0]
    try:
        data = fetch_attack(dataset=dataset, model=model, attack=attack, return_X_y=return_X_y)
    except RequestException:
        pytest.skip('Attack dataset URL down for dataset %s, model %s, and attack %s' % (dataset, model, attack))
    if return_X_y:
        assert isinstance(data, tuple)
        assert isinstance(data[0][0], np.ndarray) and isinstance(data[0][1], np.ndarray) and \
               isinstance(data[1][0], np.ndarray) and isinstance(data[1][1], np.ndarray)
    else:
        assert isinstance(data, Bunch)
        assert isinstance(data.data_train, np.ndarray) and isinstance(data.data_test, np.ndarray) and \
               isinstance(data.target_train, np.ndarray) and isinstance(data.target_test, np.ndarray)
        assert data.meta['attack_type'] == attack


# NAB dataset
files = get_list_nab()


def test_list_nab():
    assert len(files) == 58


@pytest.mark.parametrize('return_X_y', [True, False])
def test_fetch_nab(return_X_y):
    idx = np.random.choice(len(files))
    try:
        data = fetch_nab(files[idx], return_X_y=return_X_y)
    except RequestException:
        pytest.skip('NAB dataset URL down')
    if return_X_y:
        assert isinstance(data, tuple)
        assert isinstance(data[0], pd.DataFrame) and isinstance(data[1], pd.DataFrame)
    else:
        assert isinstance(data, Bunch)
        assert isinstance(data.data, pd.DataFrame) and isinstance(data.target, pd.DataFrame)


# Genome dataset
# TODO - Genome dataset is large compared to others - do we want to include in regular CI?
