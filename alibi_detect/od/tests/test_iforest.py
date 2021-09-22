from itertools import product
import pytest
from sklearn.datasets import load_iris
from alibi_detect.od import IForest
from alibi_detect.version import __version__

threshold = [None, 0.]
threshold_perc = [75., 95.]
return_instance_score = [True, False]

tests = list(product(threshold, threshold_perc, return_instance_score))
n_tests = len(tests)


@pytest.fixture
def iforest_params(request):
    threshold, threshold_perc, return_instance_score = tests[request.param]
    return threshold, threshold_perc, return_instance_score


@pytest.mark.parametrize('iforest_params', list(range(n_tests)), indirect=True)
def test_isolation_forest(iforest_params):
    threshold, threshold_perc, return_instance_score = iforest_params
    X, y = load_iris(return_X_y=True)
    iforest = IForest(threshold)
    assert iforest.threshold == threshold
    assert iforest.meta == {'name': 'IForest', 'detector_type': 'offline', 'data_type': 'tabular',
                            'version': __version__}
    iforest.fit(X)
    iforest.infer_threshold(X, threshold_perc=threshold_perc)
    iscore = iforest.score(X)
    perc_score = 100 * (iscore < iforest.threshold).astype(int).sum() / iscore.shape[0]
    assert threshold_perc + 5 > perc_score > threshold_perc - 5
    od_preds = iforest.predict(X, return_instance_score=return_instance_score)
    assert od_preds['meta'] == iforest.meta
    assert od_preds['data']['is_outlier'].sum() == (iscore > iforest.threshold).astype(int).sum()
    if not return_instance_score:
        assert od_preds['data']['instance_score'] is None
