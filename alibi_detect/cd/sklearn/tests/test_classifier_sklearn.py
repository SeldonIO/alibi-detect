import pytest
import numpy as np
from typing import Union
from alibi_detect.cd.sklearn.classifier import ClassifierDriftSklearn

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# test List[Any] inputs to the detector
def identity_fn(x: Union[np.ndarray, list]) -> np.ndarray:
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
    """Testing calibration for various models and configurations."""
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
    assert cd.meta['params']['preds_type'] == preds_type
    assert cd.meta['params']['binarize_preds '] == binarize_preds


@pytest.mark.parametrize('model', [LinearSVC(max_iter=10000),
                                   AdaBoostClassifier(),
                                   QuadraticDiscriminantAnalysis(),
                                   LogisticRegression(),
                                   GradientBoostingClassifier()])
@pytest.mark.parametrize('p_val', [0.05])
@pytest.mark.parametrize('n', [500, 1000])
@pytest.mark.parametrize('n_features', [4])
@pytest.mark.parametrize('binarize_preds', [False])
@pytest.mark.parametrize('n_folds', [2, 5])
@pytest.mark.parametrize('preds_type', ['scores'])
def test_clfdrift_scores(model, p_val, n, n_features, binarize_preds, n_folds, preds_type):
    """Testing classifier drift based on the scores (i.e. `decision_function`) for various models."""
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
    assert cd.meta['params']['preds_type'] == preds_type
    assert cd.meta['params']['binarize_preds '] == binarize_preds


@pytest.mark.parametrize('model', [SVC(probability=False), LinearSVC()])
@pytest.mark.parametrize('preds_type', ['probs'])
@pytest.mark.parametrize('use_calibration', [False])
@pytest.mark.parametrize('binarize_preds', [False])
def test_clone1(model, preds_type, use_calibration, binarize_preds):
    """
    Checks if an `AttributeError` is raised because the models do NOT support `predict_proba`,
    ``use_calibration=False`` and we are interested in the probabilities due to ``binarize_preds=False``.
    """
    with pytest.raises(AttributeError):
        ClassifierDriftSklearn(x_ref=np.random.randn(100, 5),
                               model=model,
                               preds_type=preds_type,
                               use_calibration=use_calibration,
                               binarize_preds=binarize_preds)


@pytest.mark.parametrize('model', [SVC(probability=False),
                                   LinearSVC(),
                                   LogisticRegression(),
                                   DecisionTreeClassifier(),
                                   RandomForestClassifier(),
                                   AdaBoostClassifier(),
                                   GaussianNB(),
                                   QuadraticDiscriminantAnalysis()])
@pytest.mark.parametrize('preds_type', ['probs'])
@pytest.mark.parametrize('use_calibration', [False])
@pytest.mark.parametrize('binarize_preds', [True])
def test_clone2(model, preds_type, use_calibration, binarize_preds):
    """Checks if no error is raised because ``binarize_preds=True`` and we only need access to the `predict` method."""
    ClassifierDriftSklearn(x_ref=np.random.randn(100, 5),
                           model=model,
                           preds_type=preds_type,
                           use_calibration=use_calibration,
                           binarize_preds=binarize_preds)


@pytest.mark.parametrize('model', [SVC(probability=False),
                                   LinearSVC(),
                                   LogisticRegression(),
                                   DecisionTreeClassifier(),
                                   RandomForestClassifier(),
                                   AdaBoostClassifier(),
                                   GaussianNB(),
                                   QuadraticDiscriminantAnalysis()])
@pytest.mark.parametrize('preds_type', ['probs'])
@pytest.mark.parametrize('use_calibration', [True])
@pytest.mark.parametrize('binarize_preds', [False, True])
def test_clone3(model, preds_type, use_calibration, binarize_preds):
    """
    Checks if NO error is raised because of the ``use_calibration=True`` which makes possible ``preds_types='probs'``.
    """
    ClassifierDriftSklearn(x_ref=np.random.randn(100, 5),
                           model=model,
                           preds_type=preds_type,
                           use_calibration=use_calibration,
                           binarize_preds=binarize_preds)


@pytest.mark.parametrize('model', [DecisionTreeClassifier(),
                                   RandomForestClassifier(),
                                   KNeighborsClassifier(),
                                   GaussianProcessClassifier(),
                                   MLPClassifier(),
                                   GaussianNB()])
@pytest.mark.parametrize('preds_type', ['scores'])
@pytest.mark.parametrize('use_calibration', [False, True])
@pytest.mark.parametrize('binarize_preds', [False])
def test_clone4(model, preds_type, use_calibration, binarize_preds):
    """Checks if an `AttributeError` is raised because the classifiers do not support `decision_function`."""
    with pytest.raises(AttributeError):
        ClassifierDriftSklearn(x_ref=np.random.randn(100, 5),
                               model=model,
                               preds_type=preds_type,
                               use_calibration=use_calibration,
                               binarize_preds=binarize_preds)


@pytest.mark.parametrize('model', [DecisionTreeClassifier(),
                                   RandomForestClassifier(),
                                   KNeighborsClassifier(),
                                   GaussianProcessClassifier(),
                                   MLPClassifier(),
                                   GaussianNB()])
@pytest.mark.parametrize('preds_type', ['scores'])
@pytest.mark.parametrize('use_calibration', [False, True])
@pytest.mark.parametrize('binarize_preds', [True])
def test_clone5(model, preds_type, use_calibration, binarize_preds):
    """
    Checks if `ValueError` is raised because of ``binarize_preds=True`` which conflicts with ``preds_types='scores'``.
    """
    with pytest.raises(ValueError):
        ClassifierDriftSklearn(x_ref=np.random.randn(100, 5),
                               model=model,
                               preds_type=preds_type,
                               use_calibration=use_calibration,
                               binarize_preds=binarize_preds)


@pytest.mark.parametrize('model', [SVC(probability=False), LinearSVC()])
@pytest.mark.parametrize('preds_type', ['probs'])
@pytest.mark.parametrize('use_calibration', [False])
@pytest.mark.parametrize('binarize_preds', [True])
def test_predict_proba1(model, preds_type, use_calibration, binarize_preds):
    """
    Checks if the `aux_predict_proba` is set properly to the model's `predict` when ``preds_type=probs``
    and ``binarize_preds=True``.
    """
    drift_detector = ClassifierDriftSklearn(x_ref=np.random.randn(100, 5),
                                            model=model,
                                            preds_type=preds_type,
                                            use_calibration=use_calibration,
                                            binarize_preds=binarize_preds)

    # define train and test set for internal model
    x_tr, y_tr = np.random.randn(100, 5), np.random.randint(0, 2, 100)
    x_te = np.random.randn(100, 5)

    # extract and fit internal model
    internal_model = drift_detector.model
    internal_model.fit(x_tr, y_tr)

    # check if predict matches the new predict_proba
    np.testing.assert_allclose(internal_model.predict(x_te),
                               internal_model.aux_predict_proba(x_te)[:, 1])


@pytest.mark.parametrize('model', [LogisticRegression(),
                                   GradientBoostingClassifier(),
                                   AdaBoostClassifier(),
                                   QuadraticDiscriminantAnalysis()])
@pytest.mark.parametrize('pred_types', ['scores'])
@pytest.mark.parametrize('use_calibration', [False])
@pytest.mark.parametrize('binarize_preds', [False])
def test_predict_proba2(model, pred_types, use_calibration, binarize_preds):
    """
    Checks if the `aux_predict_proba` is set properly to the model's `decision_function` when ``preds_type=scores``
    and ``binarize_preds=False``.
    """
    drift_detector = ClassifierDriftSklearn(x_ref=np.random.randn(100, 5),
                                            model=model,
                                            preds_type=pred_types,
                                            use_calibration=use_calibration,
                                            binarize_preds=binarize_preds)

    # define train and test set for internal model
    x_tr, y_tr = np.random.randn(100, 5), np.random.randint(0, 2, 100)
    x_te = np.random.randn(100, 5)

    # extract and fit internal model
    internal_model = drift_detector.model
    internal_model.fit(x_tr, y_tr)

    # check if predict matches the new predict_proba
    np.testing.assert_allclose(internal_model.decision_function(x_te),
                               internal_model.aux_predict_proba(x_te)[:, 1])


@pytest.mark.parametrize('model', [RandomForestClassifier(n_estimators=100)])
@pytest.mark.parametrize('p_val', [0.05])
@pytest.mark.parametrize('n', [500, 1000])
@pytest.mark.parametrize('n_features', [4])
@pytest.mark.parametrize('n_folds', [2, 5])
@pytest.mark.parametrize('preds_type', ['probs'])
@pytest.mark.parametrize('binarize_preds, use_calibration, use_oob', [(False, False, False),
                                                                      (False, False, True),
                                                                      (False, True, False),
                                                                      (True, False, False),
                                                                      (True, False, True),
                                                                      (True, True, False)])
def test_rf_oob(model, p_val, n, n_features, n_folds, preds_type, binarize_preds, use_calibration, use_oob):
    """
    Checks if all valid combination of `binarize_preds`, `use_calibration` and `use_oob` when ``preds_type='probs'``
    are working properly.
    """
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
        use_calibration=use_calibration,
        use_oob=use_oob
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
    assert cd.meta['params']['preds_type'] == preds_type
    assert cd.meta['params']['binarize_preds '] == binarize_preds


@pytest.mark.parametrize('model', [LogisticRegression(),
                                   GradientBoostingClassifier(),
                                   AdaBoostClassifier()])
@pytest.mark.parametrize('preds_type', ['probs'])
@pytest.mark.parametrize('use_oob', [True])
def test_clone_rf1(model, preds_type, use_oob):
    """Check if `ValueError` is raised because ``use_oob=True`` and the model is not a RandomForest."""
    with pytest.raises(ValueError):
        ClassifierDriftSklearn(x_ref=np.random.randn(100, 5),
                               model=model,
                               preds_type=preds_type,
                               use_oob=use_oob)


@pytest.mark.parametrize('model', [RandomForestClassifier()])
@pytest.mark.parametrize('preds_type', ['probs'])
@pytest.mark.parametrize('use_calibration', [False, True])
@pytest.mark.parametrize('use_oob', [True])
def test_clone_rf2(model, preds_type, use_calibration, use_oob):
    """
    Checks that ``use_oob=True`` works when ``preds_type='probs'`` for a RandomForest model with or
    without calibration.
    """
    cd = ClassifierDriftSklearn(x_ref=np.random.randn(100, 5),
                                model=model,
                                preds_type=preds_type,
                                use_calibration=use_calibration,
                                use_oob=use_oob)

    assert cd.model.oob_score
    assert cd.model.bootstrap
    assert not cd.use_calibration   # should be set to `False` when `use_oob=True`


@pytest.mark.parametrize('model', [RandomForestClassifier(oob_score=True),
                                   RandomForestClassifier(oob_score=False)])
@pytest.mark.parametrize('preds_type', ['probs'])
@pytest.mark.parametrize('use_oob', [False])
def test_clone_rf3(model, preds_type, use_oob):
    """Checks if the `oob_score` is set automatically to ``False`` when ``use_oob=False``."""
    cd = ClassifierDriftSklearn(x_ref=np.random.randn(100, 5),
                                model=model,
                                preds_type=preds_type,
                                use_oob=use_oob)
    assert not cd.model.oob_score
