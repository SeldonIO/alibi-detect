from itertools import product
import numpy as np
import pytest
import torch.nn as nn
import torch
from sklearn.linear_model import LogisticRegression
from alibi_detect.cd.preprocess import classifier_uncertainty, regressor_uncertainty

n, n_features = 100, 10
shape = (n_features,)
X_train = np.random.rand(n * n_features).reshape(n, n_features).astype('float32')
y_train_reg = np.random.rand(n).astype('float32')
y_train_clf = np.random.choice(2, n)
X_test = np.random.rand(n * n_features).reshape(n, n_features).astype('float32')


preds_type = ['probs', 'logits']
uncertainty_type = ['entropy', 'margin']
tests_cu = list(product(preds_type, uncertainty_type))
n_tests_cu = len(tests_cu)


@pytest.fixture
def cu_params(request):
    return tests_cu[request.param]


@pytest.mark.parametrize('cu_params', list(range(n_tests_cu)), indirect=True)
def test_classifier_uncertainty(cu_params):
    preds_type, uncertainty_type = cu_params
    clf = LogisticRegression().fit(X_train, y_train_clf)
    model_fn = clf.predict_log_proba if preds_type == 'logits' else clf.predict_proba
    uncertainties = classifier_uncertainty(
        X_test, model_fn, preds_type=preds_type, uncertainty_type=uncertainty_type
    )
    assert uncertainties.shape == (X_test.shape[0], 1)


tests_ru = ['mc_dropout', 'ensemble']
n_tests_ru = len(tests_ru)


@pytest.fixture
def ru_params(request):
    return tests_ru[request.param]


@pytest.mark.parametrize('ru_params', list(range(n_tests_ru)), indirect=True)
def test_regressor_uncertainty(ru_params):
    uncertainty_type = ru_params
    if uncertainty_type == 'dropout':
        model = nn.Sequential(
            nn.Linear(n_features, 10),
            nn.Dropout(0.5),
            nn.Linear(10, 1)
        )
    else:
        model = nn.Linear(n_features, 42)

    def model_fn(x):
        with torch.no_grad():
            return np.array(model(torch.as_tensor(x)))

    uncertainties = regressor_uncertainty(
        X_test, model_fn, uncertainty_type=uncertainty_type
    )
    assert uncertainties.shape == (X_test.shape[0], 1)
