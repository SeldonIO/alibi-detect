import numpy as np
import pytest
from functools import partial
from alibi_detect.cd import FETDriftOnline
from alibi_detect.utils._random import fixed_seed

n = 250
n_inits, n_reps = 3, 100
n_bootstraps = 1000
ert = 150
window_sizes = [40]

alternatives = ['less', 'greater']
n_features = [1, 3]


@pytest.mark.parametrize('alternative', alternatives)
@pytest.mark.parametrize('n_feat', n_features)
def test_fetdriftonline(alternative, n_feat, seed):
    # Reference data
    p_h0 = 0.5
    with fixed_seed(seed):
        # squeeze to test vector input in 1D case
        x_ref = np.random.choice((0, 1), (n, n_feat), p=[1 - p_h0, p_h0]).squeeze()
        x_h0 = partial(np.random.choice, (0, 1), size=n_feat, p=[1-p_h0, p_h0])

    detection_times_h0 = []
    detection_times_h1 = []
    for _ in range(n_inits):
        # Instantiate detector
        with fixed_seed(seed+1):
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
                cd.reset_state()

        # Drifted data
        if alternative == 'less':
            p_h1 = 0.1
            x_h1 = partial(np.random.choice, (0, 1), size=n_feat, p=[1-p_h1, p_h1])
        else:
            p_h1 = 0.9
            x_h1 = partial(np.random.choice, (0, 1), size=n_feat, p=[1-p_h1, p_h1])

        cd.reset_state()
        count = 0
        while len(detection_times_h1) < n_reps and count < int(1e6):
            count += 1
            x_t = x_h1().reshape(1, 1) if n_feat == 1 else x_h1()  # test shape (1,1) in 1D case here
            pred_t = cd.predict(x_t)
            if pred_t['data']['is_drift']:
                detection_times_h1.append(pred_t['data']['time'])
                cd.reset_state()

    art = np.array(detection_times_h0).mean() - np.min(window_sizes) + 1
    add = np.array(detection_times_h1).mean() - np.min(window_sizes)

    assert ert / 3 < art < 3 * ert
    assert add + 1 < ert/2


@pytest.mark.parametrize('n_feat', n_features)
def test_fet_online_state_online(n_feat, tmp_path, seed):
    """
    Test save/load/reset state methods for FETDriftOnline. State is saved, reset, and loaded, with
    prediction results and stateful attributes compared to original.
    """
    p_h0 = 0.5
    p_h1 = 0.3
    with fixed_seed(seed):
        # squeeze to test vector input in 1D case
        x_ref = np.random.choice((0, 1), (n, n_feat), p=[1 - p_h0, p_h0]).squeeze()
        x = np.random.choice((0, 1), (n, n_feat), p=[1 - p_h1, p_h1])
        dd = FETDriftOnline(x_ref, window_sizes=window_sizes, ert=20)
    # Store state for comparison
    state_dict_t0 = {}
    for key in dd.online_state_keys:
        state_dict_t0[key] = getattr(dd, key)

    # Run for 10 time steps
    test_stats_1 = []
    for t, x_t in enumerate(x):
        if t == 5:
            dd.save_state(tmp_path)
            # Store state for comparison
            state_dict_t5 = {}
            for key in dd.online_state_keys:
                state_dict_t5[key] = getattr(dd, key)
        preds = dd.predict(x_t)
        test_stats_1.append(preds['data']['test_stat'])

    # Reset and check state cleared
    dd.reset_state()
    for key, orig_val in state_dict_t0.items():
        np.testing.assert_array_equal(orig_val, getattr(dd, key))  # use np.testing here as it handles torch.Tensor etc

    # Repeat, check that same test_stats both times
    test_stats_2 = []
    for t, x_t in enumerate(x):
        preds = dd.predict(x_t)
        test_stats_2.append(preds['data']['test_stat'])
    np.testing.assert_array_equal(test_stats_1, test_stats_2)

    # Load state from t=5 timestep
    dd.load_state(tmp_path)

    # Compare stateful attributes to original at t=5
    for key, orig_val in state_dict_t5.items():
        np.testing.assert_array_equal(orig_val, getattr(dd, key))  # use np.testing here as it handles torch.Tensor etc

    # Compare predictions to original at t=5
    new_pred = dd.predict(x[5])
    np.testing.assert_array_equal(new_pred['data']['test_stat'], test_stats_1[5])
