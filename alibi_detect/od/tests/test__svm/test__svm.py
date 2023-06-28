import pytest
import numpy as np
import torch

from alibi_detect.od._svm import SVM
from alibi_detect.exceptions import NotFittedError
from alibi_detect.utils.pytorch import GaussianRBF
from sklearn.datasets import make_moons


@pytest.mark.parametrize('optimization', ['sgd', 'gd'])
def test_unfitted_svm_score(optimization):
    """Test SVM detector raises exceptions when not fitted."""
    svm_detector = SVM(
        n_components=10,
        backend='pytorch',
        kernel=GaussianRBF(torch.tensor(2)),
        optimization=optimization,
        nu=0.1
    )
    x = np.array([[0, 10], [0.1, 0]])
    x_ref = np.random.randn(100, 2)

    with pytest.raises(NotFittedError) as err:
        svm_detector.infer_threshold(x_ref, 0.1)
    assert str(err.value) == 'SVM has not been fit!'

    with pytest.raises(NotFittedError) as err:
        svm_detector.score(x)
    assert str(err.value) == 'SVM has not been fit!'

    # test predict raises exception when not fitted
    with pytest.raises(NotFittedError) as err:
        svm_detector.predict(x)
    assert str(err.value) == 'SVM has not been fit!'


@pytest.mark.parametrize('optimization,device', [('sgd', 'gpu'), ('gd', 'cpu')])
def test_svm_device_warnings(optimization, device):
    """Test SVM detector device warnings."""

    warning_msgs = {
        'sgd': ('The `sgd` optimization option is best suited for CPU. '
                'If you want to use GPU, consider using the `gd` option.'),
        'gd': ('The `gd` optimization option is best suited for GPU. '
               'If you want to use CPU, consider using the `sgd` option.')
    }

    with pytest.warns(UserWarning) as warning:
        _ = SVM(
            n_components=10,
            backend='pytorch',
            kernel=GaussianRBF(torch.tensor(2)),
            optimization=optimization,
            device=device,
            nu=0.1
        )

    assert len(warning) == 1
    assert str(warning[0].message) == warning_msgs[optimization]


def test_svm_optimization_error():
    """Test SVM detector raises correct errors for wrong optimization kwargs."""

    with pytest.raises(ValueError) as err:
        _ = SVM(
            n_components=10,
            backend='pytorch',
            kernel=GaussianRBF(torch.tensor(2)),
            optimization='not_an_option',
            device='cpu',
            nu=0.1
        )

    assert str(err.value) == 'Optimization not_an_option not recognized. Choose from `sgd` or `gd`.'


@pytest.mark.parametrize('optimization,score_bounds', [('sgd', [-0.15, -0.6]), ('gd', [-0.85, -0.9])])
def test_fitted_svm_score(optimization, score_bounds):
    """Test SVM detector score method.

    Test SVM detector that has been fitted on reference data but has not had a threshold
    inferred can still score data using the predict method. Test that it does not raise an error
    but does not return `threshold`, `p_value` and `is_outlier` values.
    """
    svm_detector = SVM(
        n_components=10,
        backend='pytorch',
        kernel=GaussianRBF(torch.tensor(2)),
        optimization=optimization,
        nu=0.1
    )
    x_ref = np.random.randn(100, 2)
    svm_detector.fit(x_ref)
    x = np.array([[0, 10], [0.1, 0]])
    scores = svm_detector.score(x)

    y = svm_detector.predict(x)
    y = y['data']
    assert y['instance_score'][0] > score_bounds[0]
    assert y['instance_score'][1] < score_bounds[1]
    assert all(y['instance_score'] == scores)
    assert not y['threshold_inferred']
    assert y['threshold'] is None
    assert y['is_outlier'] is None
    assert y['p_value'] is None


@pytest.mark.parametrize('optimization,score_bounds', [('sgd', [-0.15, -0.6]), ('gd', [-0.85, -0.9])])
def test_fitted_svm_predict(optimization, score_bounds):
    """Test SVM detector predict method.

    Test SVM detector that has been fitted on reference data and has had a threshold
    inferred can score data using the predict method as well as predict outliers. Test that it
    returns `threshold`, `p_value` and `is_outlier` values.
    """
    svm_detector = SVM(
        n_components=10,
        backend='pytorch',
        kernel=GaussianRBF(torch.tensor(2)),
        optimization=optimization,
        nu=0.1
    )
    x_ref = np.random.randn(100, 2)
    svm_detector.fit(x_ref)
    svm_detector.infer_threshold(x_ref, 0.1)
    x = np.array([[0, 10], [0, 0.1]])
    y = svm_detector.predict(x)
    y = y['data']
    assert y['instance_score'][0] > score_bounds[0]
    assert y['instance_score'][1] < score_bounds[1]
    assert y['threshold_inferred']
    assert y['threshold'] is not None
    assert y['p_value'].all()
    assert (y['is_outlier'] == [True, False]).all()


@pytest.mark.parametrize('optimization', ['sgd', 'gd'])
@pytest.mark.parametrize('n_components', [None, 100])
def test_svm_integration(optimization, n_components):
    """Test SVM detector on moons dataset.

    Test SVM detector on a more complex 2d example. Test that the detector can be fitted
    on reference data and infer a threshold. Test that it differentiates between inliers and outliers.
    """
    svm_detector = SVM(
        n_components=n_components,
        nu=0.1,
        backend='pytorch',
        kernel=GaussianRBF(torch.tensor(2)),
        optimization=optimization,
    )
    X_ref, _ = make_moons(1001, shuffle=True, noise=0.05, random_state=None)
    X_ref, x_inlier = X_ref[0:1000], X_ref[1000][None]
    svm_detector.fit(X_ref)
    svm_detector.infer_threshold(X_ref, 0.1)
    result = svm_detector.predict(x_inlier)
    result = result['data']['is_outlier'][0]
    assert not result

    x_outlier = np.array([[-1, 1.5]])
    result = svm_detector.predict(x_outlier)
    result = result['data']['is_outlier'][0]
    assert result


@pytest.mark.skip(reason="Can't convert default kernel GaussianRBF to torchscript due to torchscript type constraints")
def test_svm_torchscript(tmp_path):
    """Tests user can torch-script svm detector."""
    sigma = torch.tensor(0.2)
    svm_detector = SVM(
        n_components=100,
        backend='pytorch',
        kernel=GaussianRBF(sigma=sigma)
    )
    X_ref, _ = make_moons(1001, shuffle=True, noise=0.05, random_state=None)
    X_ref, x_inlier = X_ref[0:1000], X_ref[1000][None]
    svm_detector.fit(X_ref, nu=0.1)
    svm_detector.infer_threshold(X_ref, 0.1)
    x_outlier = np.array([[-1, 1.5]])
    x = torch.tensor([x_inlier[0], x_outlier[0]], dtype=torch.float32)

    ts_svm = torch.jit.script(svm_detector.backend)
    y = ts_svm(x)
    assert torch.all(y == torch.tensor([False, True]))

    ts_svm.save(tmp_path / 'svm.pt')
    ts_svm = torch.load(tmp_path / 'svm.pt')
    y = ts_svm(x)
    assert torch.all(y == torch.tensor([False, True]))