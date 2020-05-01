from functools import reduce
from itertools import product
from operator import mul
import numpy as np
import pytest
from alibi_detect.utils.data import Bunch
from alibi_detect.utils.perturbation import apply_mask, inject_outlier_ts, mutate_categorical

x = np.random.rand(20 * 20 * 3).reshape(1, 20, 20, 3)
mask_size = [(2, 2), (8, 8)]
n_masks = [1, 10]
channels = [[0], [0, 1, 2]]
mask_type = ['uniform', 'normal', 'zero']
clip_rng = [(0, 1), (.25, .75)]

tests = list(product(mask_size, n_masks, channels, mask_type, clip_rng))
n_tests = len(tests)


@pytest.fixture
def apply_mask_params(request):
    mask_size, n_masks, channels, mask_type, clip_rng = tests[request.param]
    return mask_size, n_masks, channels, mask_type, clip_rng


@pytest.mark.parametrize('apply_mask_params', list(range(n_tests)), indirect=True)
def test_apply_mask(apply_mask_params):
    mask_size, n_masks, channels, mask_type, clip_rng = apply_mask_params
    X_mask, mask = apply_mask(x,
                              mask_size=mask_size,
                              n_masks=n_masks,
                              channels=channels,
                              mask_type=mask_type,
                              clip_rng=clip_rng
                              )
    assert X_mask.shape[0] == mask.shape[0] == n_masks
    total_masked = n_masks * mask_size[0] * mask_size[1] * len(channels)
    if mask_type == 'zero':
        assert (mask == X_mask).astype(int).sum() == total_masked
    else:
        assert clip_rng[0] <= X_mask.min() and clip_rng[1] >= X_mask.max()
        assert (X_mask == np.clip(x + mask, clip_rng[0], clip_rng[1])).astype(int).sum() \
            == reduce(mul, list(x.shape)) * n_masks


N = 1000
x_ts = [np.random.rand(N).reshape(-1, 1), np.random.rand(3 * N).reshape(-1, 3)]
perc_outlier = [0, 10, 20]
min_std = [0, 1]

tests_ts = list(product(x_ts, perc_outlier, min_std))
n_tests_ts = len(tests_ts)


@pytest.fixture
def inject_outlier_ts_params(request):
    return tests_ts[request.param]


@pytest.mark.parametrize('inject_outlier_ts_params', list(range(n_tests_ts)), indirect=True)
def test_inject_outlier_ts(inject_outlier_ts_params):
    X, perc_outlier, min_std = inject_outlier_ts_params
    data = inject_outlier_ts(X, perc_outlier, perc_window=10, n_std=2., min_std=min_std)
    assert isinstance(data, Bunch)
    X_outlier, is_outlier = data.data, data.target
    assert X_outlier.shape[0] == N == is_outlier.shape[0]
    assert perc_outlier - 5 < is_outlier.mean() * 100 < perc_outlier + 5
    X_diff = (X_outlier != X).astype(int).sum(axis=1)
    idx_diff = np.where(X_diff != 0)[0]
    idx_outlier = np.where(is_outlier != 0)[0]
    if perc_outlier > 0:
        assert (idx_diff == idx_outlier).all()
    else:
        assert not idx_diff and not idx_outlier


rate = [0., .1, .2]
x_mutate = [np.zeros(10000), np.zeros((10, 10, 10, 1))]
feature_range = [(0, 1), (0, 2)]
tests_mutate = list(product(rate, x_mutate, feature_range))
n_tests_mutate = len(tests_mutate)


@pytest.fixture
def mutate_params(request):
    return tests_mutate[request.param]


@pytest.mark.parametrize('mutate_params', list(range(n_tests_mutate)), indirect=True)
def test_mutate(mutate_params):
    rate, x_mutate, feature_range = mutate_params
    x_pert = mutate_categorical(x_mutate, rate, feature_range=feature_range).numpy()
    delta = ((x_mutate - x_pert) != 0).astype(int)
    eps = rate * .5
    assert rate - eps <= delta.sum() / np.prod(x_mutate.shape) <= rate + eps
    if rate > 0.:
        assert x_pert.min() == feature_range[0] and x_pert.max() == feature_range[1]
