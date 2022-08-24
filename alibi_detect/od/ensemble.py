import imp
from typing import Optional, List
from alibi_detect.od.base import OutlierDetector
from alibi_detect.od.aggregation import BaseTransform, AverageAggregator, PValNormaliser
from alibi_detect.od.processor import BaseProcessor


class Ensemble(OutlierDetector):
    def __init__(
            self,
            detectors: List[OutlierDetector], 
            aggregator: Optional[BaseTransform] = AverageAggregator(), 
            normaliser: Optional[BaseTransform] = PValNormaliser(),
            processor=BaseProcessor):

        self.detectors = detectors
        self.normaliser = normaliser
        self.aggregator = aggregator
        self.processor = processor(detectors=detectors)

    def fit(self, X):
        for detector in self.detectors:
            detector.fit(X)

    def score(self, X):
        return self.processor(X)
