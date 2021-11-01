from itertools import product
import numpy as np
import pytest
from alibi_detect.od import SpectralResidual
from alibi_detect.version import __version__

# create normal time series and one with perturbations
t = np.linspace(0, 0.5, 1000)
X = np.sin(40 * 2 * np.pi * t) + 0.5 * np.sin(90 * 2 * np.pi * t)
idx_pert = np.random.randint(0, 1000, 10)
X_pert = X.copy()
X_pert[idx_pert] = 10

window_amp = [10, 20]
window_local = [20, 30]
n_est_points = [10, 20]
return_instance_score = [True, False]

tests = list(product(window_amp, window_local, n_est_points, return_instance_score))
n_tests = len(tests)


@pytest.fixture
def sr_params(request):
    return tests[request.param]


@pytest.mark.parametrize('sr_params', list(range(n_tests)), indirect=True)
def test_sr(sr_params):
    window_amp, window_local, n_est_points, return_instance_score = sr_params

    threshold = 2.5
    od = SpectralResidual(threshold=threshold, window_amp=window_amp,
                          window_local=window_local, n_est_points=n_est_points)

    assert od.threshold == threshold
    assert od.meta == {'name': 'SpectralResidual',
                       'detector_type': 'online',
                       'data_type': 'time-series',
                       'version': __version__}
    preds_in = od.predict(X, t, return_instance_score=return_instance_score)
    assert preds_in['data']['is_outlier'].sum() <= 2.
    if return_instance_score:
        assert preds_in['data']['is_outlier'].sum() == (preds_in['data']['instance_score']
                                                        > od.threshold).astype(int).sum()
    else:
        assert preds_in['data']['instance_score'] is None
    preds_out = od.predict(X_pert, t, return_instance_score=return_instance_score)
    assert preds_out['data']['is_outlier'].sum() > 0
    if return_instance_score:
        assert preds_out['data']['is_outlier'].sum() == (preds_out['data']['instance_score']
                                                         > od.threshold).astype(int).sum()
    else:
        assert preds_out['data']['instance_score'] is None
    assert preds_out['meta'] == od.meta
