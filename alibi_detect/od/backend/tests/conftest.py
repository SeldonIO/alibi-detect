import pytest
from alibi_detect.od.backend.torch.ensemble import Accumulator, PValNormaliser, AverageAggregator


@pytest.fixture(scope='session')
def accumulator(request):
    return Accumulator(
        normaliser=PValNormaliser(),
        aggregator=AverageAggregator()
    )
