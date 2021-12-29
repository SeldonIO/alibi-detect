import pytest
import numpy as np
from typing import Union
from alibi_detect.cd.sklearn.classifier import ClassifierDriftSklearn

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# test List[Any] inputs to the detector
def identity_fn(x: Union[np.array, list]) -> np.array:
    if isinstance(x, list):
        return np.array(x)
    else:
        return x


@pytest.mark.parametrize('model, use_calibration, calibration_kwargs', [
    (LogisticRegression(max_iter=10000), False, None),
    (SVC(max_iter=10000, probability=True), False, None),
    (LinearSVC(max_iter=10000), True, {'method': 'sigmoid'}),
    (LinearSVC(max_iter=10000), True, {'method': 'isotonic'}),
    (DecisionTreeClassifier(), False, None),
    (RandomForestClassifier(n_estimators=50), False, None),
    (GradientBoostingClassifier(n_estimators=50), False, None)
])
@pytest.mark.parametrize('preds_type', ['probs'])
@pytest.mark.parametrize('p_val', [0.05])
@pytest.mark.parametrize('n', [1000])
@pytest.mark.parametrize('n_features', [4])
@pytest.mark.parametrize('binarize_preds', [True, False])
@pytest.mark.parametrize('n_folds', [None, 2])
@pytest.mark.parametrize('train_size', [0.5])
@pytest.mark.parametrize('preprocess_batch', [None, identity_fn])
@pytest.mark.parametrize('update_x_ref', [{'last': 1000}, {'reservoir_sampling': 1000}])
def test_clfdrift_calibration(model, preds_type, p_val, n, n_features, binarize_preds, n_folds, train_size,
                              preprocess_batch, update_x_ref, use_calibration, calibration_kwargs):
    np.random.seed(0)

    x_ref = np.random.randn(n, n_features)
    x_test0 = np.random.randn(n, n_features)
    x_test1 = np.random.randn(n, n_features) + 1

    to_list = False
    if preprocess_batch is not None:
        to_list = True
        x_ref = [_ for _ in x_ref]
        update_x_ref = None

    cd = ClassifierDriftSklearn(
        x_ref=x_ref,
        model=model,
        preds_type=preds_type,
        p_val=p_val,
        update_x_ref=update_x_ref,
        train_size=train_size,
        n_folds=n_folds,
        binarize_preds=binarize_preds,
        use_calibration=use_calibration,
        calibration_kwargs=calibration_kwargs
    )

    if to_list:
        x_test0 = [_ for _ in x_test0]
    preds_0 = cd.predict(x_test0)
    assert cd.n == len(x_test0) + len(x_ref)
    assert preds_0['data']['is_drift'] == 0
    assert preds_0['data']['distance'] >= 0

    if to_list:
        x_test1 = [_ for _ in x_test1]
    preds_1 = cd.predict(x_test1)
    assert cd.n == len(x_test1) + len(x_test0) + len(x_ref)
    assert preds_1['data']['is_drift'] == 1
    assert preds_1['data']['distance'] >= 0

    assert preds_0['data']['distance'] < preds_1['data']['distance']
    assert cd.meta['params']['preds_type'] == 'probs'
    assert cd.meta['params']['binarize_preds '] == binarize_preds


@pytest.mark.parametrize('model', [LinearSVC(max_iter=10000)])
@pytest.mark.parametrize('p_val', [0.05])
@pytest.mark.parametrize('n', [500, 1000])
@pytest.mark.parametrize('n_features', [4])
@pytest.mark.parametrize('binarize_preds', [False])
@pytest.mark.parametrize('n_folds', [2, 5])
@pytest.mark.parametrize('preds_type', ['scores'])
def test_clfdrift_scores(model, p_val, n, n_features, binarize_preds, n_folds, preds_type):
    np.random.seed(0)

    x_ref = np.random.randn(n, n_features)
    x_test0 = np.random.randn(n, n_features)
    x_test1 = np.random.randn(n, n_features) + 1
    cd = ClassifierDriftSklearn(
        x_ref=x_ref,
        preds_type=preds_type,
        model=model,
        p_val=p_val,
        n_folds=n_folds,
        binarize_preds=binarize_preds,
    )

    preds_0 = cd.predict(x_test0)
    assert cd.n == len(x_test0) + len(x_ref)
    assert preds_0['data']['is_drift'] == 0
    assert preds_0['data']['distance'] >= 0

    preds_1 = cd.predict(x_test1)
    assert cd.n == len(x_test1) + len(x_test0) + len(x_ref)
    assert preds_1['data']['is_drift'] == 1
    assert preds_1['data']['distance'] >= 0

    assert preds_0['data']['distance'] < preds_1['data']['distance']
    assert cd.meta['params']['preds_type'] == 'scores'
    assert cd.meta['params']['binarize_preds '] == binarize_preds