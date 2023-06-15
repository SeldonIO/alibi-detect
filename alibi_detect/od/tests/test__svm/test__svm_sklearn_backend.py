import pytest
import numpy as np

from alibi_detect.od.sklearn.svm import SVMSklearn
from alibi_detect.exceptions import NotFittedError, ThresholdNotInferredError


@pytest.mark.parametrize('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
def test_svm_sklearn_scoring(kernel):
    """Test sklearn SVM detector scoring method.

    Tests the scoring method of the sklearn SVM backend detector.
    """
    svm_sklearn = SVMSklearn(
        n_components=100,
        kernel='rbf'
    )
    mean = [8, 8]
    cov = [[2., 0.], [0., 1.]]
    x_ref = np.random.multivariate_normal(mean, cov, 1000)
    svm_sklearn.fit(x_ref, nu=0.01)

    x_1 = np.array([[8., 8.]])
    scores_1 = svm_sklearn.score(x_1)

    x_2 = np.array([[13., 13.]])
    scores_2 = svm_sklearn.score(x_2)

    x_3 = np.array([[-100., 100.]])
    scores_3 = svm_sklearn.score(x_3)

    # test correct ordering of scores given relative outlyingness of data
    assert scores_1 < scores_2 < scores_3

    # test that detector correctly detects true outlier
    svm_sklearn.infer_threshold(x_ref, 0.01)
    x = np.concatenate((x_1, x_2, x_3))
    outputs = svm_sklearn.predict(x)
    assert np.all(outputs.is_outlier == np.array([False, True, True]))
    assert np.all(svm_sklearn(x) == np.array([False, True, True]))

    # test that 0.01 of the in distribution data is flagged as outliers
    x = np.random.multivariate_normal(mean, cov, 1000)
    outputs = svm_sklearn.predict(x)
    assert (outputs.is_outlier.sum()/1000) - 0.01 < 0.01


def test_svm_sklearn_backend_fit_errors():
    """Test SVM detector sklearn backend fit errors.

    Tests the correct errors are raised when using the SVM sklearn backend detector.
    """
    svm_sklearn = SVMSklearn(n_components=100, kernel='rbf')
    assert not svm_sklearn.fitted

    # Test that the backend raises an error if it is not fitted before
    # calling forward method.
    x = np.random.randn(1, 10)
    with pytest.raises(NotFittedError) as err:
        svm_sklearn(x)
    assert str(err.value) == 'SVMSklearn has not been fit!'

    # Test that the backend raises an error if it is not fitted before
    # predicting.
    with pytest.raises(NotFittedError) as err:
        svm_sklearn.predict(x)
    assert str(err.value) == 'SVMSklearn has not been fit!'

    # Test the backend updates _fitted flag on fit.
    x_ref = np.random.randn(1024, 10)
    svm_sklearn.fit(x_ref, nu=0.01)
    assert svm_sklearn.fitted

    # Test that the backend raises an if the forward method is called without the
    # threshold being inferred.
    with pytest.raises(ThresholdNotInferredError) as err:
        svm_sklearn(x)
    assert str(err.value) == 'SVMSklearn has no threshold set, call `infer_threshold` to fit one!'

    # Test that the backend can call predict without the threshold being inferred.
    assert svm_sklearn.predict(x)


@pytest.mark.parametrize('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
def test_svm_sklearn_fit(kernel):
    """Test SVM detector sklearn fit method.

    Tests sklearn detector checks for convergence and stops early if it does.
    """
    svm_sklearn = SVMSklearn(n_components=1, kernel=kernel)
    mean = [8, 8]
    cov = [[2., 0.], [0., 1.]]
    x_ref = np.random.multivariate_normal(mean, cov, 1000)
    fit_results = svm_sklearn.fit(x_ref, tol=0.01, nu=0.01)
    assert fit_results['converged']
    assert fit_results['n_iter'] < 100
