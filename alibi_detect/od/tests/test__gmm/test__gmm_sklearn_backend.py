import pytest
import numpy as np

from alibi_detect.od.sklearn.gmm import GMMSklearn
from alibi_detect.base import NotFitException, ThresholdNotInferredException


def test_gmm_sklearn_backend_fit_errors():
    gmm_sklearn = GMMSklearn(n_components=2)
    assert not gmm_sklearn._fitted

    # Test that the backend raises an error if it is not fitted before
    # calling forward method.
    x = np.random.randn(1, 10)
    with pytest.raises(NotFitException) as err:
        gmm_sklearn(x)
    assert str(err.value) == 'GMMSklearn has not been fit!'

    # Test that the backend raises an error if it is not fitted before
    # predicting.
    with pytest.raises(NotFitException) as err:
        gmm_sklearn.predict(x)
    assert str(err.value) == 'GMMSklearn has not been fit!'

    # Test the backend updates _fitted flag on fit.
    x_ref = np.random.randn(1024, 10)
    gmm_sklearn.fit(x_ref)
    assert gmm_sklearn._fitted

    # Test that the backend raises an if the forward method is called without the
    # threshold being inferred.
    with pytest.raises(ThresholdNotInferredException) as err:
        gmm_sklearn(x)
    assert str(err.value) == 'GMMSklearn has no threshold set, call `infer_threshold` before predicting.'

    # Test that the backend can call predict without the threshold being inferred.
    assert gmm_sklearn.predict(x)


def test_gmm_sklearn_scoring():
    gmm_sklearn = GMMSklearn(n_components=2)
    mean = [8, 8]
    cov = [[2., 0.], [0., 1.]]
    x_ref = np.random.multivariate_normal(mean, cov, 1000)
    gmm_sklearn.fit(x_ref)

    x_1 = np.array([[8., 8.]])
    scores_1 = gmm_sklearn.score(x_1)

    x_2 = np.random.multivariate_normal(mean, cov, 1)
    scores_2 = gmm_sklearn.score(x_2)

    x_3 = np.array([[-10., 10.]])
    scores_3 = gmm_sklearn.score(x_3)

    # test correct ordering of scores given outlyingness of data
    assert scores_1 < scores_2 < scores_3

    # test that detector correctly detects true Outlier
    gmm_sklearn.infer_threshold(x_ref, 0.01)
    x = np.concatenate((x_1, x_2, x_3))
    outputs = gmm_sklearn.predict(x)
    assert np.all(outputs.is_outlier == np.array([False, False, True]))
    assert np.all(gmm_sklearn(x) == np.array([False, False, True]))

    # test that 0.01 of the in distribution data is flagged as outliers
    x = np.random.multivariate_normal(mean, cov, 1000)
    outputs = gmm_sklearn.predict(x)
    assert (outputs.is_outlier.sum()/1000) - 0.01 < 0.01
