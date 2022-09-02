from typing import Optional, List
from alibi_detect.od.base import OutlierDetector
from alibi_detect.od.aggregation import BaseTransform, AverageAggregator, PValNormaliser
from alibi_detect.od.processor import BaseProcessor
from alibi_detect.od.config import ConfigMixin
from alibi_detect.saving.registry import registry


@registry.register('Ensemble')
class Ensemble(OutlierDetector, ConfigMixin):
    CONFIG_PARAMS = ('detectors', 'aggregator', 'normaliser', 'processor')
    LARGE_PARAMS = ('detectors', )
    BASE_OBJ = True

    def __init__(
            self,
            detectors: List[OutlierDetector], 
            aggregator: Optional[BaseTransform] = AverageAggregator(), 
            normaliser: Optional[BaseTransform] = PValNormaliser(),
            processor=BaseProcessor()):
        self._set_config(locals())

        self.detectors = detectors
        self.normaliser = normaliser
        self.aggregator = aggregator
        self.processor = processor

    def fit(self, X):
        for detector in self.detectors:
            detector.fit(X)

    def score(self, X):
        return self.processor(X, self.detectors)
