from itertools import product
import numpy as np
import pytest
from sklearn.datasets import load_iris
from alibi_detect.od import Mahalanobis
from alibi_detect.version import __version__

threshold = [None, 5.]
n_components = [2, 3]
std_clip = [2, 3]
start_clip = [10, 1000]
max_n = [None, 50]
threshold_perc = [75., 95.]
return_instance_score = [True, False]

tests = list(product(threshold, n_components, std_clip, start_clip,
                     max_n, threshold_perc, return_instance_score))
n_tests = len(tests)


@pytest.fixture
def mahalanobis_params(request):
    return tests[request.param]


@pytest.mark.parametrize('mahalanobis_params', list(range(n_tests)), indirect=True)
def test_mahalanobis(mahalanobis_params):
    threshold, n_components, std_clip, start_clip, max_n, \
        threshold_perc, return_instance_score = mahalanobis_params
    X, y = load_iris(return_X_y=True)
    mh = Mahalanobis(threshold, n_components=n_components, std_clip=std_clip,
                     start_clip=start_clip, max_n=max_n)
    assert mh.threshold == threshold
    assert mh.n == 0
    assert mh.meta == {'name': 'Mahalanobis', 'detector_type': 'online', 'data_type': 'tabular', 'version': __version__}
    mh.infer_threshold(X, threshold_perc=threshold_perc)
    assert mh.n == X.shape[0]
    iscore = mh.score(X)  # noqa
    assert mh.n == 2 * X.shape[0]
    assert mh.mean.shape[0] == X.shape[1]
    assert mh.C.shape == (X.shape[1], X.shape[1])
    assert (np.diag(mh.C) >= 0).all()
    od_preds = mh.predict(X, return_instance_score=return_instance_score)
    assert mh.n == 3 * X.shape[0]
    assert od_preds['meta'] == mh.meta
    if return_instance_score:
        assert od_preds['data']['is_outlier'].sum() == (od_preds['data']['instance_score']
                                                        > mh.threshold).astype(int).sum()
    else:
        assert od_preds['data']['instance_score'] is None
