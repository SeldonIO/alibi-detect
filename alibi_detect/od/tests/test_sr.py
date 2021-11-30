import pytest
import numpy as np
from alibi_detect.od import SpectralResidual
from alibi_detect.version import __version__


@pytest.fixture(scope='module')
def signal():
    np.random.seed(0)

    # create normal time series and one with perturbations
    t = np.linspace(0, 0.5, 1000)
    X = np.sin(40 * 2 * np.pi * t) + 0.5 * np.sin(90 * 2 * np.pi * t)

    idx_pert = np.random.choice(np.arange(1000), size=10, replace=False)  # ensure we perturb exactly 10 points
    X_pert = X.copy()
    X_pert[idx_pert] = 10
    return {"t": t, "X": X, "X_pert": X_pert}


@pytest.mark.parametrize('window_amp', [10, 20])
@pytest.mark.parametrize('window_local', [20, 30])
@pytest.mark.parametrize('n_est_points', [10, 20])
@pytest.mark.parametrize('return_instance_score', [True, False])
def test_detector(signal, window_amp, window_local, n_est_points, return_instance_score):
    t, X, X_pert = signal["t"], signal['X'], signal['X_pert']

    threshold = 6
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
    assert preds_out['data']['is_outlier'].sum() >= 10  # check if we detect at least the number of perturbed points
    if return_instance_score:
        assert preds_out['data']['is_outlier'].sum() == (preds_out['data']['instance_score']
                                                         > od.threshold).astype(int).sum()
    else:
        assert preds_out['data']['instance_score'] is None
    assert preds_out['meta'] == od.meta


@pytest.mark.parametrize('method', ['constant', 'replicate', 'reflect'])
@pytest.mark.parametrize('side', ['left', 'right', 'bilateral'])
def test_padding(method, side):
    np.random.seed(0)

    for _ in range(100):
        X_size = np.random.randint(low=10, high=1000)
        W_size = np.random.randint(low=2, high=X_size - 1)

        X = np.random.randint(low=0, high=10, size=X_size)
        W = np.random.randint(low=0, high=10, size=W_size)

        X_pad = SpectralResidual.pad_same(X=X, W=W, method=method, side=side)
        X_conv = np.convolve(X_pad, W, 'valid')
        assert X_conv.shape[0] == X_size

        # length of the padding for laterals
        pad_right = (W_size - 1) // 2
        pad_left = W_size - 1 - pad_right

        if method == 'constant':
            if side == 'left':
                assert np.all(X_pad[:W_size - 1] == 0)
            elif side == 'right':
                assert np.all(X_pad[-W_size + 1:] == 0)
            else:
                if pad_left > 0:
                    assert np.all(X_pad[:pad_left] == 0)
                if pad_right > 0:
                    assert np.all(X_pad[-pad_right:] == 0)

        elif method == 'replicate':
            if side == 'left':
                assert np.all(X_pad[:W_size - 1] == X[0])
            elif side == 'right':
                assert np.all(X_pad[-W_size + 1:] == X[-1])
            else:
                if pad_left > 0:
                    assert np.all(X_pad[:pad_left] == X[0])
                if pad_right > 0:
                    assert np.all(X_pad[-pad_right:] == X[-1])
        else:
            if side == 'left':
                assert np.all(X_pad[:W_size - 1] == X[1:W_size][::-1])
            elif side == 'right':
                assert np.all(X_pad[-W_size + 1:] == X[-2:-W_size - 1:-1])
            else:
                if pad_left > 0:
                    assert np.all(X_pad[:pad_left] == X[1:pad_left + 1][::-1])
                if pad_right > 0:
                    assert np.all(X_pad[-pad_right:] == X[-pad_right - 1:-1][::-1])
