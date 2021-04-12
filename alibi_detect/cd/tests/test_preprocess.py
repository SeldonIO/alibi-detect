import numpy as np
import pytest
from alibi_detect.cd.preprocess import pca

n, n_features = 100, 10
shape = (n_features,)
X = np.random.rand(n * n_features).reshape(n, n_features).astype('float32')

tests_pca = [2, 4]
n_tests_pca = len(tests_pca)


@pytest.fixture
def pca_params(request):
    return tests_pca[request.param]


@pytest.mark.parametrize('pca_params', list(range(n_tests_pca)), indirect=True)
def test_pca(pca_params):
    n_components = pca_params
    X_pca = pca(X, n_components)
    assert X_pca.shape[-1] == n_components
