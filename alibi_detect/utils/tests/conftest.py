import pytest
from alibi_testing.data import get_adult_data, get_iris_data, get_movie_sentiment_data
import alibi_testing


@pytest.fixture(scope='module')
def adult_data():
    return get_adult_data()


@pytest.fixture(scope='module')
def iris_data():
    return get_iris_data()


@pytest.fixture(scope='module')
def movie_sentiment_data():
    return get_movie_sentiment_data()


@pytest.fixture(scope='module')
def mnist_predictor():
    model = alibi_testing.load('mnist-cnn-tf2.2.0')
    predictor = lambda x: model.predict(x)  # noqa
    return predictor
