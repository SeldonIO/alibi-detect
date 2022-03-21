import numpy as np
from pytest_cases import parametrize
from alibi_testing.data import get_movie_sentiment_data

# Note: If any of below cases become large, see https://smarie.github.io/python-pytest-cases/#c-caching-cases


# Group dataset "cases" by type of data i.e. continuous, binary, categorical, mixed
class ContinuousData:
    # Note: we could parametrize cases here (and/or pass them fixtures).
    #  See https://smarie.github.io/python-pytest-cases/#case-generators
    @staticmethod
    @parametrize(data_shape=[(50, 4)])
    def data_synthetic_nd(data_shape):
        n_samples, input_dim = data_shape
        X_ref = np.random.default_rng(0).normal(0, 0.5, size=data_shape)
        X_h0 = np.random.default_rng(1).normal(0, 0.5, size=data_shape)
        return X_ref, X_h0

#    @staticmethod
#    def data_synthetic_1d():  # TODO - add if we decide to support 1D data
#        n_samples = 50
#        X_ref = np.random.rand(n_samples)
#        X_h0 = np.random.rand(n_samples)
#        return X_ref, X_h0


class CategoricalData:
    @staticmethod
    @parametrize(data_shape=[(50, 4)])
    def data_synthetic_nd(data_shape):
        n_samples, input_dim = data_shape
        X_ref = np.random.default_rng(0).choice(a=[0, 1, 2], size=(n_samples, input_dim), p=[0.5, 0.3, 0.2])
        X_h0 = np.random.default_rng(1).choice(a=[0, 1, 2], size=(n_samples, input_dim), p=[0.5, 0.3, 0.2])
        return X_ref, X_h0


class MixedData:
    @staticmethod
    @parametrize(data_shape=[(50, 4)])
    def data_synthetic_nd(data_shape):
        n_samples, input_dim = data_shape
        X_ref = np.random.default_rng(0).normal(0, 0.5, size=data_shape)
        X_ref[:, :2] = np.random.default_rng(0).choice(a=[0, 1, 2], size=(n_samples, 2), p=[0.5, 0.3, 0.2])
        X_h0 = np.random.default_rng(1).normal(0, 0.5, size=data_shape)
        X_h0[:, :2] = np.random.default_rng(1).choice(a=[0, 1, 2], size=(n_samples, 2), p=[0.5, 0.3, 0.2])
        return X_ref, X_h0


class BinData:
    @staticmethod
    @parametrize(data_shape=[(50, 2)])
    def data_synthetic_nd(data_shape):
        n_samples, input_dim = data_shape
        X_ref = np.random.default_rng(0).choice([0, 1], (n_samples, input_dim), p=[0.6, 0.4])
        X_h0 = np.random.default_rng(0).choice([0, 1], (n_samples, input_dim), p=[0.6, 0.4])
        return X_ref, X_h0


class TextData:
    @staticmethod
    def movie_sentiment_data():
        return get_movie_sentiment_data()
