import numpy as np
import pytest
from itertools import product
from alibi_detect.cd import CVMDriftOnline

n, n_test = 200, 500
n_bootstraps = 1000
ert = 50
np.random.seed(0)

window_sizes = [[10], [10, 20]]
batch_size = [None, int(n_bootstraps/4)]
n_features = [1, 3]

tests_cvmdriftonline = list(product(window_sizes, batch_size, n_features))
n_tests = len(tests_cvmdriftonline)


@pytest.fixture
def cvmdriftonline_params(request):
    return tests_cvmdriftonline[request.param]


@pytest.mark.parametrize('cvmdriftonline_params', list(range(n_tests)), indirect=True)
def test_cvmdriftonline(cvmdriftonline_params):
    window_sizes, batch_size, n_feat = cvmdriftonline_params

    # Reference data
    x_ref = np.random.normal(0, 1, size=(n, n_feat)).squeeze()  # squeeze to test vec input in 1D case

    # Instantiate detector
    cd = CVMDriftOnline(x_ref=x_ref, ert=ert, window_sizes=window_sizes,
                        n_bootstraps=n_bootstraps, batch_size=batch_size)

    # Test predict
    x_h0 = np.random.normal(0, 1, size=(n_test, n_feat))
    x_h1 = np.random.normal(1, 1, size=(n_test, n_feat))

    # Reference data
    detection_times_h0 = []
    test_stats_h0 = []
    for x_t in x_h0:  # x_t is np.int64 in 1D, np.ndarray in multi-D
        t0 = cd.t
        pred_t = cd.predict(x_t, return_test_stat=True)
        assert cd.t - t0 == 1  # This checks state updated (self.t at least)
        test_stats_h0.append(pred_t['data']['test_stat'])
        if pred_t['data']['is_drift']:
            detection_times_h0.append(pred_t['data']['time'])
            cd.reset()
    art = np.array(detection_times_h0).mean() - np.min(window_sizes) + 1
    test_stats_h0 = [ts for ts in test_stats_h0 if ts is not None]
    assert ert/3 < art < 3*ert

    # Drifted data
    cd.reset()
    detection_times_h1 = []
    test_stats_h1 = []
    for x_t in x_h1:
        pred_t = cd.predict(x_t, return_test_stat=True)
        test_stats_h1.append(pred_t['data']['test_stat'])
        if pred_t['data']['is_drift']:
            detection_times_h1.append(pred_t['data']['time'])
            cd.reset()
    add = np.array(detection_times_h1).mean() - np.min(window_sizes)
    test_stats_h1 = [ts for ts in test_stats_h1 if ts is not None]
    assert add < ert/2

    assert np.nanmean(test_stats_h1) > np.nanmean(test_stats_h0)
