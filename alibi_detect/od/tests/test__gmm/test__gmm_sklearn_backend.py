import pytest
import numpy as np

from alibi_detect.od.sklearn.gmm import GMMSklearn
from alibi_detect.od.base import NotFitException, ThresholdNotInferredException


def test_gmm_sklearn_backend_fit_errors():
    gm_sklearn = GMMSklearn(n_components=2)
    assert not gm_sklearn._fitted

    x = np.random.randn(1, 10)
    with pytest.raises(NotFitException) as err:
        gm_sklearn(x)
    assert str(err.value) == 'GMMSklearn has not been fit!'

    with pytest.raises(NotFitException) as err:
        gm_sklearn.predict(x)
    assert str(err.value) == 'GMMSklearn has not been fit!'

    x_ref = np.random.randn(1024, 10)
    gm_sklearn.fit(x_ref)

    assert gm_sklearn._fitted

    with pytest.raises(ThresholdNotInferredException) as err:
        gm_sklearn(x)
    assert str(err.value) == 'GMMSklearn has no threshold set, call `infer_threshold` before predicting.'

    assert gm_sklearn.predict(x)


def test_gmm_linear_scoring():
    gm_sklearn = GMMSklearn(n_components=2)
    mean = [8, 8]
    cov = [[2., 0.], [0., 1.]]
    x_ref = np.random.multivariate_normal(mean, cov, 1000)
    gm_sklearn.fit(x_ref)

    x_1 = np.array([[8., 8.]])
    scores_1 = gm_sklearn.score(x_1)

    x_2 = np.random.multivariate_normal(mean, cov, 1)
    scores_2 = gm_sklearn.score(x_2)

    x_3 = np.array([[-10., 10.]])
    scores_3 = gm_sklearn.score(x_3)

    # test correct ordering of scores given outlyingness of data
    assert scores_1 < scores_2 < scores_3

    # test that detector correctly detects true Outlier
    gm_sklearn.infer_threshold(x_ref, 0.01)
    x = np.concatenate((x_1, x_2, x_3))
    outputs = gm_sklearn.predict(x)
    assert np.all(outputs.is_outlier == np.array([False, False, True]))
    assert np.all(gm_sklearn(x) == np.array([False, False, True]))

    # test that 0.01 of the in distribution data is flagged as outliers
    x = np.random.multivariate_normal(mean, cov, 1000)
    outputs = gm_sklearn.predict(x)
    assert (outputs.is_outlier.sum()/1000) - 0.01 < 0.01
