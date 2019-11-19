from functools import reduce
from itertools import product
from operator import mul
import numpy as np
import pytest
from alibi_detect.utils.perturbation import apply_mask

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
