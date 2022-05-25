import pytest


@pytest.fixture
def seed(pytestconfig):
    """
    Returns the random seed set by pytest-randomly.
    """
    return pytestconfig.getoption("randomly_seed")
