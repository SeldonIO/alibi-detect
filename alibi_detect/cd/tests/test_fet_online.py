import numpy as np
import pytest
from functools import partial
from alibi_detect.cd import FETDriftOnline

STATE_DICT = ('t', 'xs')
n = 250
n_inits, n_reps = 3, 100
n_bootstraps = 1000
ert = 150
window_sizes = [40]

alternatives = ['less', 'greater']
n_features = [1, 3]


@pytest.mark.parametrize('alternative', alternatives)
@pytest.mark.parametrize('n_feat', n_features)
def test_fetdriftonline(alternative, n_feat):
    # Reference data
    np.random.seed(0)
    p_h0 = 0.5
    x_ref = np.random.choice((0, 1), (n, n_feat), p=[1 - p_h0, p_h0]).squeeze()  # squeeze to test vec input in 1D case
    x_h0 = partial(np.random.choice, (0, 1), size=n_feat, p=[1-p_h0, p_h0])

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
            x_t = int(x_h0()) if n_feat == 1 else x_h0()  # x_t is int in 1D case, otherwise ndarray with shape (n_feat)
            t0 = cd.t
            pred_t = cd.predict(x_t)
            assert cd.t - t0 == 1  # This checks state updated (self.t at least)
            if pred_t['data']['is_drift']:
                detection_times_h0.append(pred_t['data']['time'])
                cd.reset()

        # Drifted data
        if alternative == 'less':
            p_h1 = 0.1
            x_h1 = partial(np.random.choice, (0, 1), size=n_feat, p=[1-p_h1, p_h1])
        else:
            p_h1 = 0.9
            x_h1 = partial(np.random.choice, (0, 1), size=n_feat, p=[1-p_h1, p_h1])

        cd.reset()
        count = 0
        while len(detection_times_h1) < n_reps and count < int(1e6):
            count += 1
            x_t = x_h1().reshape(1, 1) if n_feat == 1 else x_h1()  # test shape (1,1) in 1D case here
            pred_t = cd.predict(x_t)
            if pred_t['data']['is_drift']:
                detection_times_h1.append(pred_t['data']['time'])
                cd.reset()

    art = np.array(detection_times_h0).mean() - np.min(window_sizes) + 1
    add = np.array(detection_times_h1).mean() - np.min(window_sizes)

    assert ert / 3 < art < 3 * ert
    assert add + 1 < ert/2


@pytest.mark.parametrize('n_feat', n_features)
def test_fet_online_state_functional(n_feat, tmp_path):
    """
    A functional test of save/load/reset methods or FETDriftOnline. State is saved, reset, and loaded, with
    prediction results checked.
    """
    p_h0 = 0.5
    p_h1 = 0.3
    x_ref = np.random.choice((0, 1), (n, n_feat), p=[1 - p_h0, p_h0]).squeeze()  # squeeze to test vec input in 1D case
    x = np.random.choice((0, 1), (n, n_feat), p=[1 - p_h1, p_h1])  # squeeze to test vec input in 1D case
    dd = FETDriftOnline(x_ref, window_sizes=window_sizes, ert=20)

    # Run for 50 time steps
    test_stats_1 = []
    for t, x_t in enumerate(x):
        preds = dd.predict(x_t)
        test_stats_1.append(preds['data']['test_stat'])
        if t == 20:
            dd.save_state(tmp_path)

    # Clear state and repeat, check that same test_stats both times
    dd.reset()
    test_stats_2 = []
    for t, x_t in enumerate(x):
        preds = dd.predict(x_t)
        test_stats_2.append(preds['data']['test_stat'])
    np.testing.assert_array_equal(test_stats_1, test_stats_2)

    # Load state from t=20 timestep and check results of t=21 the same
    dd.load_state(tmp_path)
    new_pred = dd.predict(x[21])
    np.testing.assert_array_equal(new_pred['data']['test_stat'], test_stats_1[21])


@pytest.mark.parametrize('n_feat', n_features)
def test_fet_online_state_unit(n_feat, tmp_path):
    """
    A unit-type test of save/load/reset methods or FETDriftOnline. Stateful attributes in STATE_DICT are
    compared pre and post save/load.
    """
    p_h0 = 0.5
    x_ref = np.random.choice((0, 1), (n, n_feat), p=[1 - p_h0, p_h0]).squeeze()  # squeeze to test vec input in 1D case
    dd = FETDriftOnline(x_ref, window_sizes=[10], ert=20)
    # Get original state
    orig_state_dict = {}
    for key in STATE_DICT:
        orig_state_dict[key] = getattr(dd, key)
    # Save, reset and load
    dd.save_state(tmp_path)
    dd.reset()
    dd.load_state(tmp_path)
    # Compare state to original
    for key, orig_val in orig_state_dict.items():
        np.testing.assert_array_equal(orig_val, getattr(dd, key))  # use np.testing here as it handles tt.Tensor etc
