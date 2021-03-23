from itertools import product
import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
import tensorflow as tf
from alibi_detect.cd import MarginDensityDrift

margin = [0.2]
density_range = [(0.005, 0.10)]
tests = list(product(margin, density_range))
n_tests = len(tests)

# load breast cancer data
X, y = load_breast_cancer(return_X_y=True)
X = X.astype(np.float32)

input_dim = X.shape[1]

# define and train model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(input_dim,)),
    tf.keras.layers.Dense(16, activation=tf.nn.relu),
    tf.keras.layers.Dense(16, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
])
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X, y, batch_size=128, epochs=20, verbose=False)


@pytest.fixture
def margindrift_params(request):
    return tests[request.param]


@pytest.mark.parametrize('margindrift_params', list(range(n_tests)), indirect=True)
def test_margindrift(margindrift_params):
    margin, density_range = margindrift_params

    np.random.seed(0)
    tf.random.set_seed(0)

    # init MarginDensityDrift
    md3 = MarginDensityDrift(
        margin=margin,
        model=model,
        density_range=density_range
    )

    assert md3.margin == margin
    assert md3.density_range == density_range
    assert md3.meta == {'name': 'MarginDensityDrift', 'detector_type': 'offline', 'data_type': None}

    X_test0 = X.copy()
    X_test1 = np.ones_like(X_test0)

    preds_0 = md3.predict(X_test0)
    assert preds_0['data']['margin_density'] == md3.score(X_test0)
    assert preds_0['data']['is_drift'] == 0

    preds_1 = md3.predict(X_test1)
    assert preds_1['data']['margin_density'] == md3.score(X_test1)
    assert preds_1['data']['is_drift'] == 1
