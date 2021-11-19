import numpy as np
import pytest
from functools import partial
from alibi_detect.cd import FETDriftOnline

n, n_features = 250, 1
n_inits, n_reps = 3, 100
n_bootstraps = 1000
ert = 150
window_sizes = [40]

alternatives = ['less', 'greater']

tests_fetdriftonline = list(alternatives)
n_tests = len(tests_fetdriftonline)


@pytest.fixture
def fetdriftonline_params(request):
    return tests_fetdriftonline[request.param]


@pytest.mark.parametrize('fetdriftonline_params', list(range(n_tests)), indirect=True)
def test_fetdriftonline(fetdriftonline_params):
    alternative = fetdriftonline_params

    # Reference data
    np.random.seed(0)
    p_h0 = 0.5
    x_ref = np.random.choice(2, (n, n_features), p=[1 - p_h0, p_h0])
    x_h0 = partial(np.random.choice, (0, 1), size=(1, n_features), p=[1-p_h0, p_h0])

    detection_times_h0 = []
    detection_times_h1 = []
    for init in range(n_inits):
        # Instantiate detector
        np.random.seed(init+1)
        cd = FETDriftOnline(x_ref=x_ref, ert=ert, window_sizes=window_sizes,
                            n_bootstraps=n_bootstraps, alternative=alternative)

        # Reference data
        count = 0
        while len(detection_times_h0) < n_reps and count < int(1e6):
            count += 1
            x_t = x_h0()
            t0 = cd.t
            pred_t = cd.predict(x_t)
            assert cd.t - t0 == 1  # This checks state updated (self.t at least)
            if pred_t['data']['is_drift']:
                detection_times_h0.append(pred_t['data']['time'])
                cd.reset()

        # Drifted data
        if alternative == 'less':
            p_h1 = 0.1
            x_h1 = partial(np.random.choice, (0, 1), size=(1, n_features), p=[1-p_h1, p_h1])
        else:
            p_h1 = 0.9
            x_h1 = partial(np.random.choice, (0, 1), size=(1, n_features), p=[1-p_h1, p_h1])

        cd.reset()
        count = 0
        while len(detection_times_h1) < n_reps and count < int(1e6):
            count += 1
            x_t = x_h1()
            pred_t = cd.predict(x_t)
            if pred_t['data']['is_drift']:
                detection_times_h1.append(pred_t['data']['time'])
                cd.reset()

    art = np.array(detection_times_h0).mean() - np.min(window_sizes) + 1
    add = np.array(detection_times_h1).mean() - np.min(window_sizes)

    assert ert / 3 < art < 3 * ert
    assert add + 1 < ert/2
