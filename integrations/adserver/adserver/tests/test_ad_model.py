from alibi_detect.base import BaseDetector
import numpy as np
from unittest import TestCase
from adserver.ad_model import AlibiDetectAdversarialDetectionModel, HEADER_RETURN_INSTANCE_SCORE
from typing import Dict

FAKE_RESULT = {"adversarial": 1}


class DummyAEModel(BaseDetector):
    def __init__(self, expected_return_instance_score: bool= True):
        super().__init__()
        self.expected_return_instance_score = expected_return_instance_score

    def score(self, X: np.ndarray):
        return FAKE_RESULT

    def predict(
        self,
        X: np.ndarray,
        batch_size: int = int(1e10),
        return_instance_score: bool = True,
    ) -> Dict[Dict[str, str], Dict[str, np.ndarray]]:
        assert return_instance_score == self.expected_return_instance_score
        return FAKE_RESULT


class TestAEModel(TestCase):
    def test_basic(self):
        model = DummyAEModel()
        ad_model = AlibiDetectAdversarialDetectionModel(
            "name", "s3://model", model=model
        )
        req = [1, 2]
        headers = {}
        res = ad_model.process_event(req, headers)
        self.assertEqual(res, FAKE_RESULT)

    def test_no_return_instance_score(self):
        model = DummyAEModel(expected_return_instance_score=False)
        ad_model = AlibiDetectAdversarialDetectionModel(
            "name", "s3://model", model=model
        )
        req = [1, 2]
        headers = {HEADER_RETURN_INSTANCE_SCORE: "false"}
        res = ad_model.process_event(req, headers)
        self.assertEqual(res, FAKE_RESULT)