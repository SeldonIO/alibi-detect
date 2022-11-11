from itertools import product
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from alibi_detect.utils.tensorflow import GaussianRBF, DeepKernel, BaseKernel, RationalQuadratic, Periodic, \
    log_sigma_median
from alibi_detect.utils.tensorflow.distance import squared_pairwise_distance

sigma = [None, np.array([1.]), np.array([1., 2.])]
n_features = [5, 10]
n_instances = [(100, 100), (100, 75)]
trainable = [True, False]
tests_gk = list(product(sigma, n_features, n_instances, trainable))
n_tests_gk = len(tests_gk)


@pytest.fixture
def gaussian_kernel_params(request):
    return tests_gk[request.param]


@pytest.mark.parametrize('gaussian_kernel_params', list(range(n_tests_gk)), indirect=True)
def test_gaussian_kernel(gaussian_kernel_params):
    sigma, n_features, n_instances, trainable = gaussian_kernel_params
    xshape, yshape = (n_instances[0], n_features), (n_instances[1], n_features)
    x = tf.convert_to_tensor(np.random.random(xshape).astype('float32'))
    y = tf.convert_to_tensor(np.random.random(yshape).astype('float32'))

    kernel = GaussianRBF(sigma=sigma, trainable=trainable)
    infer_parameter = True if sigma is None else False
    if trainable and infer_parameter:
        with pytest.raises(Exception):
            kernel(x, y, infer_parameter=infer_parameter)
    else:
        k_xy = kernel(x, y, infer_parameter=infer_parameter).numpy()
        k_xx = kernel(x, x, infer_parameter=infer_parameter).numpy()
        assert k_xy.shape == n_instances and k_xx.shape == (xshape[0], xshape[0])
        np.testing.assert_almost_equal(k_xx.trace(), xshape[0], decimal=4)
        assert (k_xx > 0.).all() and (k_xy > 0.).all()


def log_sigma_mean(x: tf.Tensor, y: tf.Tensor, dist: tf.Tensor) -> tf.Tensor:
    sigma = tf.expand_dims(.5 * tf.reduce_mean(tf.reshape(dist, (-1,))) ** .5, axis=0)
    return tf.math.log(sigma)


kernel_ref = ['GaussianRBF', 'RationalQuadratic', 'Periodic']
n_features = [5, 10]
n_instances = [(100, 100), (100, 75)]
trainable = [True, False]
init_fn = [None, log_sigma_median, log_sigma_mean]
tests_init_fn = list(product(kernel_ref, n_features, n_instances, trainable, init_fn))


@pytest.fixture
def init_fn_params(request):
    return tests_init_fn[request.param]


@pytest.mark.parametrize('init_fn_params', list(range(len(tests_init_fn))), indirect=True)
def test_init_fn(init_fn_params):
    kernel_ref, n_features, n_instances, trainable, init_fn = init_fn_params
    xshape, yshape = (n_instances[0], n_features), (n_instances[1], n_features)
    x = tf.convert_to_tensor(np.random.random(xshape).astype('float32'))
    y = tf.convert_to_tensor(np.random.random(yshape).astype('float32'))

    if kernel_ref == 'GaussianRBF':
        kernel = GaussianRBF(trainable=trainable, init_fn_sigma=init_fn)
    elif kernel_ref == 'RationalQuadratic':
        kernel = RationalQuadratic(trainable=trainable, init_fn_sigma=init_fn)
    elif kernel_ref == 'Periodic':
        kernel = Periodic(trainable=trainable, init_fn_sigma=init_fn)
    else:
        raise NotImplementedError
    if trainable:
        with pytest.raises(Exception):
            kernel(x, y, infer_parameter=True)
    else:
        k_xy = kernel(x, y, infer_parameter=True).numpy()
        k_xx = kernel(x, x, infer_parameter=True).numpy()
        assert k_xy.shape == n_instances and k_xx.shape == (xshape[0], xshape[0])
        np.testing.assert_almost_equal(k_xx.trace(), xshape[0], decimal=4)
        assert (k_xx > 0.).all() and (k_xy > 0.).all()
        if init_fn is not None:
            np.testing.assert_almost_equal(kernel.sigma.numpy(),
                                           np.exp(init_fn(x, y, squared_pairwise_distance(x, y)).numpy()),
                                           decimal=4)


sigma = [None, np.array([1.]), np.array([2.])]
alpha = [None, np.array([1.]), np.array([2.])]
n_features = [5, 10]
n_instances = [(100, 100), (100, 75)]
trainable = [True, False]
tests_rqk = list(product(sigma, alpha, n_features, n_instances, trainable))
n_tests_rqk = len(tests_rqk)


@pytest.fixture
def rationalquadratic_kernel_params(request):
    return tests_rqk[request.param]


@pytest.mark.parametrize('rationalquadratic_kernel_params', list(range(n_tests_rqk)), indirect=True)
def test_rationalquadratic_kernel(rationalquadratic_kernel_params):
    sigma, alpha, n_features, n_instances, trainable = rationalquadratic_kernel_params
    xshape, yshape = (n_instances[0], n_features), (n_instances[1], n_features)
    x = tf.convert_to_tensor(np.random.random(xshape).astype('float32'))
    y = tf.convert_to_tensor(np.random.random(yshape).astype('float32'))

    kernel = RationalQuadratic(sigma=sigma, alpha=alpha, trainable=trainable)
    infer_parameter = True if sigma is None else False
    if trainable and infer_parameter:
        with pytest.raises(Exception):
            kernel(x, y, infer_parameter=infer_parameter)
    else:
        k_xy = kernel(x, y, infer_parameter=infer_parameter).numpy()
        k_xx = kernel(x, x, infer_parameter=infer_parameter).numpy()
        assert k_xy.shape == n_instances and k_xx.shape == (xshape[0], xshape[0])
        np.testing.assert_almost_equal(k_xx.trace(), xshape[0], decimal=4)
        assert (k_xx > 0.).all() and (k_xy > 0.).all()


sigma = [None, np.array([1.]), np.array([2.])]
tau = [None, np.array([8.]), np.array([24.])]
n_features = [5, 10]
n_instances = [(100, 100), (100, 75)]
trainable = [True, False]
tests_pk = list(product(sigma, tau, n_features, n_instances, trainable))
n_tests_pk = len(tests_pk)


@pytest.fixture
def periodic_kernel_params(request):
    return tests_pk[request.param]


@pytest.mark.parametrize('periodic_kernel_params', list(range(n_tests_pk)), indirect=True)
def test_periodic_kernel(periodic_kernel_params):
    sigma, tau, n_features, n_instances, trainable = periodic_kernel_params
    xshape, yshape = (n_instances[0], n_features), (n_instances[1], n_features)
    x = tf.convert_to_tensor(np.random.random(xshape).astype('float32'))
    y = tf.convert_to_tensor(np.random.random(yshape).astype('float32'))

    kernel = Periodic(sigma=sigma, tau=tau, trainable=trainable)
    infer_parameter = True if sigma is None else False
    if trainable and infer_parameter:
        with pytest.raises(Exception):
            kernel(x, y, infer_parameter=infer_parameter)
    else:
        k_xy = kernel(x, y, infer_parameter=infer_parameter).numpy()
        k_xx = kernel(x, x, infer_parameter=infer_parameter).numpy()
        assert k_xy.shape == n_instances and k_xx.shape == (xshape[0], xshape[0])
        np.testing.assert_almost_equal(k_xx.trace(), xshape[0], decimal=4)
        assert (k_xx > 0.).all() and (k_xy > 0.).all()


sigma_0 = [None, np.array([1.])]
sigma_1 = [None, np.array([1.])]
sigma_2 = [None, np.array([1.])]
operation_0 = ['*', '+']
operation_1 = ['*', '+']
n_features = [5, 10]
n_instances = [(100, 100), (100, 75)]
trainable = [True, False]
tests_ck = list(product(sigma_0, sigma_1, sigma_2,
                        operation_0, operation_1, n_features, n_instances, trainable))
n_tests_ck = len(tests_ck)


@pytest.fixture
def comp_kernel_params(request):
    return tests_ck[request.param]


@pytest.mark.parametrize('comp_kernel_params', list(range(n_tests_ck)), indirect=True)
def test_comp_kernel(comp_kernel_params):
    (sigma_0, sigma_1, sigma_2, operation_0, operation_1,
     n_features, n_instances, trainable) = comp_kernel_params
    xshape, yshape = (n_instances[0], n_features), (n_instances[1], n_features)
    x = tf.convert_to_tensor(np.random.random(xshape).astype('float32'))
    y = tf.convert_to_tensor(np.random.random(yshape).astype('float32'))

    kernel_0 = GaussianRBF(sigma=sigma_0, trainable=trainable)
    kernel_1 = GaussianRBF(sigma=sigma_1, trainable=trainable)
    kernel_2 = GaussianRBF(sigma=sigma_2, trainable=trainable)
    if operation_0 == '*' and operation_1 == '*':
        kernel = kernel_0 * kernel_1 * kernel_2
    elif operation_0 == '*' and operation_1 == '+':
        kernel = (kernel_0 * kernel_1 + kernel_2) / tf.convert_to_tensor(2.0)  # ensure k(x, x) = 1
    elif operation_0 == '+' and operation_1 == '*':
        kernel = (kernel_0 + kernel_1 * kernel_2) / tf.convert_to_tensor(2.0)  # ensure k(x, x) = 1
    elif operation_0 == '+' and operation_1 == '+':
        kernel = (kernel_0 + kernel_1 + kernel_2) / tf.convert_to_tensor(3.0)  # ensure k(x, x) = 1
    else:
        with pytest.raises(Exception):
            raise Exception('Invalid operation')
    infer_parameter = True if sigma is None else False
    if trainable and infer_parameter:
        with pytest.raises(Exception):
            kernel(x, y, infer_parameter=infer_parameter)
    else:
        k_xy = kernel(x, y, infer_parameter=infer_parameter).numpy()
        k_xx = kernel(x, x, infer_parameter=infer_parameter).numpy()
        assert k_xy.shape == n_instances and k_xx.shape == (xshape[0], xshape[0])
        np.testing.assert_almost_equal(k_xx.trace(), xshape[0], decimal=4)
        assert (k_xx > 0.).all() and (k_xy > 0.).all()


class MyKernel(BaseKernel):  # TODO: Support then test models using keras functional API
    def __init__(self, n_features: int):
        super().__init__()
        self.dense = Dense(20)

    def call(self, x: tf.Tensor, y: tf.Tensor, infer_parameter: bool = False) -> tf.Tensor:
        return tf.einsum('ji,ki->jk', self.dense(x), self.dense(y))


n_features = [5, 10]
n_instances = [(100, 100), (100, 75)]
kernel_a = [GaussianRBF(trainable=True), MyKernel]
kernel_b = [GaussianRBF(trainable=True), MyKernel, None]
eps = [0.5, 'trainable']
tests_dk = list(product(n_features, n_instances, kernel_a, kernel_b, eps))
n_tests_dk = len(tests_dk)


@pytest.fixture
def deep_kernel_params(request):
    return tests_dk[request.param]


@pytest.mark.parametrize('deep_kernel_params', list(range(n_tests_dk)), indirect=True)
def test_deep_kernel(deep_kernel_params):
    n_features, n_instances, kernel_a, kernel_b, eps = deep_kernel_params
    xshape, yshape = (n_instances[0], n_features), (n_instances[1], n_features)
    x = tf.convert_to_tensor(np.random.random(xshape).astype('float32'))
    y = tf.convert_to_tensor(np.random.random(yshape).astype('float32'))

    proj = tf.keras.Sequential([Input(shape=(n_features,)), Dense(n_features)])
    kernel_a = kernel_a(n_features) if kernel_a == MyKernel else kernel_a
    kernel_b = kernel_b(n_features) if kernel_b == MyKernel else kernel_b

    kernel = DeepKernel(proj, kernel_a=kernel_a, kernel_b=kernel_b, eps=eps)

    k_xy = kernel(x, y).numpy()
    k_yx = kernel(y, x).numpy()
    k_xx = kernel(x, x).numpy()
    assert k_xy.shape == n_instances and k_xx.shape == (xshape[0], xshape[0])
    assert (np.diag(k_xx) > 0.).all()
    np.testing.assert_almost_equal(k_xy, np.transpose(k_yx), decimal=5)
