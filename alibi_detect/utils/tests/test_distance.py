import dask.array as da
import numpy as np
from scipy.spatial.distance import cityblock
from itertools import product
import pytest
from alibi_detect.utils.distance import (pairwise_distance, maximum_mean_discrepancy, abdm,
                                         cityblock_batch, mvdm, multidim_scaling, relative_euclidean_distance)

n_features = [2, 5]
n_instances = [(100, 100), (100, 75)]
tests_pairwise = list(product(n_features, n_instances))
n_tests_pairwise = len(tests_pairwise)


@pytest.fixture
def pairwise_params(request):
    return tests_pairwise[request.param]


@pytest.mark.parametrize('pairwise_params', list(range(n_tests_pairwise)), indirect=True)
def test_pairwise(pairwise_params):
    n_features, n_instances = pairwise_params
    xshape, yshape = (n_instances[0], n_features), (n_instances[1], n_features)
    np.random.seed(0)
    x = np.random.random(xshape).astype('float32')
    y = np.random.random(yshape).astype('float32')
    xda = da.from_array(x, chunks=xshape)
    yda = da.from_array(y, chunks=yshape)

    dist_xx = pairwise_distance(x, x)
    dist_xy = pairwise_distance(x, y)
    dist_xx_da = pairwise_distance(xda, xda).compute()
    dist_xy_da = pairwise_distance(xda, yda).compute()

    assert dist_xx.shape == dist_xx_da.shape == (xshape[0], xshape[0])
    assert dist_xy.shape == dist_xy_da.shape == n_instances
    assert (dist_xx == dist_xx_da).all() and (dist_xy == dist_xy_da).all()
    assert dist_xx.trace() == 0.


tests_mmd = tests_pairwise
n_tests_mmd = n_tests_pairwise


@pytest.fixture
def mmd_params(request):
    return tests_mmd[request.param]


@pytest.mark.parametrize('mmd_params', list(range(n_tests_mmd)), indirect=True)
def test_mmd(mmd_params):
    n_features, n_instances = mmd_params
    xshape, yshape = (n_instances[0], n_features), (n_instances[1], n_features)
    np.random.seed(0)
    x = np.random.random(xshape).astype('float32')
    y = np.random.random(yshape).astype('float32')
    xda = da.from_array(x, chunks=xshape)
    yda = da.from_array(y, chunks=yshape)

    kwargs = {'sigma': np.array([1.])}
    mmd_xx = maximum_mean_discrepancy(x, x, **kwargs)
    mmd_xy = maximum_mean_discrepancy(x, y, **kwargs)
    mmd_xx_da = maximum_mean_discrepancy(xda, xda, **kwargs).compute()
    mmd_xy_da = maximum_mean_discrepancy(xda, yda, **kwargs).compute()

    assert mmd_xx == mmd_xx_da and mmd_xy == mmd_xy_da
    assert mmd_xy > mmd_xx


dims = np.array([1, 10, 50])
shapes = list(product(dims, dims))
n_tests = len(dims) ** 2


@pytest.fixture
def random_matrix(request):
    shape = shapes[request.param]
    matrix = np.random.rand(*shape)
    return matrix


@pytest.mark.parametrize('random_matrix', list(range(n_tests)), indirect=True)
def test_cityblock_batch(random_matrix):
    X = random_matrix
    y = X[np.random.choice(X.shape[0])]

    batch_dists = cityblock_batch(X, y)
    single_dists = np.array([cityblock(x, y) for x in X]).reshape(X.shape[0], -1)

    assert np.allclose(batch_dists, single_dists)


n_cat = [2, 3, 4]
n_labels = [2, 3]
n_items = [20, 50, 100]
cols = [1, 5]
tests = list(product(n_cat, n_labels, n_items, cols))
n_tests = len(tests)


@pytest.fixture
def cats_and_labels(request):
    cat, label, items, cols = tests[request.param]
    cats = np.random.randint(0, cat, items * cols).reshape(-1, cols)
    labels = np.random.randint(0, label, items).reshape(-1, 1)
    return cats, labels


@pytest.mark.parametrize('cats_and_labels', list(range(n_tests)), indirect=True)
def test_abdm_mvdm(cats_and_labels):
    X, y = cats_and_labels
    n_cols = X.shape[1]
    cat_vars = {i: None for i in range(n_cols)}
    if n_cols > 1:
        d_pair = abdm(X, cat_vars)
    else:
        d_pair = mvdm(X, y, cat_vars)
    assert list(d_pair.keys()) == list(cat_vars.keys())
    for k, v in d_pair.items():
        assert v.shape == (cat_vars[k], cat_vars[k])
        assert v.min() >= 0


Xy = (4, 2, 100, 5)
idx = np.where([t == Xy for t in tests])[0].item()
feature_range = ((np.ones((1, 5)) * -1).astype(np.float32),
                 (np.ones((1, 5))).astype(np.float32))


@pytest.mark.parametrize('cats_and_labels,rng,update_rng,center',
                         [(idx, feature_range, False, False),
                          (idx, feature_range, True, False),
                          (idx, feature_range, False, True),
                          (idx, feature_range, True, True)],
                         indirect=['cats_and_labels'])
def test_multidim_scaling(cats_and_labels, rng, update_rng, center):
    # compute pairwise distance
    X, y = cats_and_labels
    n_cols = X.shape[1]
    cat_vars = {i: None for i in range(n_cols)}
    d_pair = abdm(X, cat_vars)

    # apply multidimensional scaling
    d_abs, new_rng = multidim_scaling(d_pair,
                                      feature_range=rng,
                                      update_feature_range=update_rng,
                                      center=center
                                      )
    assert list(d_abs.keys()) == list(cat_vars.keys())
    if update_rng:
        assert (new_rng[0] != rng[0]).any()
        assert (new_rng[1] != rng[1]).any()
    else:
        assert (new_rng[0] == rng[0]).all()
        assert (new_rng[1] == rng[1]).all()

    for k, v in d_abs.items():
        assert v.shape[0] == d_pair[k].shape[0]
        if center:
            assert (v.max() + v.min()) - (rng[1][0, k] + rng[0][0, k]) < 1e-5


def test_relative_euclidean_distance():
    x = np.random.rand(5, 3)
    y = np.random.rand(5, 3)

    assert (relative_euclidean_distance(x, y).numpy() == relative_euclidean_distance(y, x).numpy()).all()
    assert (relative_euclidean_distance(x, x).numpy() == relative_euclidean_distance(y, y).numpy()).all()
    assert (relative_euclidean_distance(x, y).numpy() >= 0.).all()
