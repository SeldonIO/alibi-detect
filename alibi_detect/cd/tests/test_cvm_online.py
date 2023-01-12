import numpy as np
import pytest
from alibi_detect.cd import CVMDriftOnline
from alibi_detect.utils._random import fixed_seed

n, n_test = 200, 500
n_bootstraps = 1000
ert = 50
np.random.seed(0)

window_sizes = [[10], [10, 20]]
batch_size = [None, int(n_bootstraps/4)]
n_features = [1, 3]


@pytest.mark.parametrize('window_sizes', window_sizes)
@pytest.mark.parametrize('batch_size', batch_size)
@pytest.mark.parametrize('n_feat', n_features)
def test_cvmdriftonline(window_sizes, batch_size, n_feat, seed):
    with fixed_seed(seed):
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
            cd.reset_state()
    art = np.array(detection_times_h0).mean() - np.min(window_sizes) + 1
    test_stats_h0 = [ts for ts in test_stats_h0 if ts is not None]
    assert ert/3 < art < 3*ert

    # Drifted data
    cd.reset_state()
    detection_times_h1 = []
    test_stats_h1 = []
    for x_t in x_h1:
        pred_t = cd.predict(x_t, return_test_stat=True)
        test_stats_h1.append(pred_t['data']['test_stat'])
        if pred_t['data']['is_drift']:
            detection_times_h1.append(pred_t['data']['time'])
            cd.reset_state()
    add = np.array(detection_times_h1).mean() - np.min(window_sizes)
    test_stats_h1 = [ts for ts in test_stats_h1 if ts is not None]
    assert add < ert/2

    assert np.nanmean(test_stats_h1) > np.nanmean(test_stats_h0)


@pytest.mark.parametrize('n_feat', n_features)
def test_cvm_online_state_online(n_feat, tmp_path, seed):
    """
    Test save/load/reset state methods for CVMDriftOnline. State is saved, reset, and loaded, with
    prediction results and stateful attributes compared to original.
    """
    window_sizes = [10]

    with fixed_seed(seed):
        x_ref = np.random.normal(0, 1, (n, n_feat)).squeeze()
        x = np.random.normal(0.1, 1, (n, n_feat))
        dd = CVMDriftOnline(x_ref, window_sizes=window_sizes, ert=20)
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
