import json
from typing import List, Dict

import ceserver
import kfserving
import numpy as np
from .numpy_encoder import NumpyEncoder
from alibi_detect.utils.saving import load_detector

EVENT_SOURCE_PREFIX = "seldon.ceserver.adserver"
EVENT_TYPE = "seldon.outlier"

HEADER_RETURN_FEATURE_SCORE = "alibi-detect-return-feature-score"
HEADER_RETURN_INSTANCE_SCORE = "alibi-detect-return-instance-score"
HEADER_OUTLIER_TYPE = "alibi-detect-outlier-type"

class AlibiDetectModel(ceserver.CEModel):  # pylint:disable=c-extension-no-member
    def __init__(self, name: str, storage_uri: str):
        """
        Outlier Detection / Concept Drift Model

        Parameters
        ----------
        name
             The name of the model
        storage_uri
             The URI location of the model
        """
        super().__init__(name)
        self.name = name
        self.storage_uri = storage_uri
        self.ready = False
        self.model = None

    def load(self):
        """
        Load the model from storage

        """
        model_folder = kfserving.Storage.download(self.storage_uri)
        self.model = load_detector(model_folder)
        self.ready = True

    def process_event(self, inputs: List, headers: Dict) -> Dict:
        """
        Process the event and return Alibi Detect score

        Parameters
        ----------
        inputs
             Input data
        headers
             Header options

        Returns
        -------
             Alibi Detect response

        """
        try:
            X = np.array(inputs)
        except Exception as e:
            raise Exception(
                "Failed to initialize NumPy array from inputs: %s, %s" % (e, inputs))

        ret_instance_score = False
        if HEADER_RETURN_INSTANCE_SCORE in headers and headers[
            HEADER_RETURN_INSTANCE_SCORE] == "true":
            ret_instance_score = True

        # Check if there are VAE outlier headers
        if HEADER_OUTLIER_TYPE in headers or HEADER_RETURN_FEATURE_SCORE in headers:
            outlier_type = 'instance'
            if HEADER_OUTLIER_TYPE in headers and headers[HEADER_OUTLIER_TYPE]:
                outlier_type = headers[HEADER_OUTLIER_TYPE]
            ret_feature_score = False
            if HEADER_RETURN_FEATURE_SCORE in headers and headers[HEADER_RETURN_FEATURE_SCORE] == "true":
                ret_feature_score = True
            od_preds = self.model.predict(X,
                                          outlier_type=outlier_type,
                                          # use 'feature' or 'instance' level
                                          return_feature_score=ret_feature_score,
                                          # scores used to determine outliers
                                          return_instance_score=ret_instance_score)
        else: # generic method
            od_preds = self.model.predict(X, return_instance_score=ret_instance_score)

        return json.loads(json.dumps(od_preds, cls=NumpyEncoder))

    def event_source(self) -> str:
        return EVENT_SOURCE_PREFIX + self.name

    def event_type(self) -> str:
        return EVENT_TYPE

    def headers(self) -> List:
        return [HEADER_OUTLIER_TYPE, HEADER_RETURN_FEATURE_SCORE, HEADER_RETURN_INSTANCE_SCORE]