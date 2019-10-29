import json
from typing import List, Dict

import ceserver
import kfserving
import numpy as np
from kfserving.protocols.util import NumpyEncoder
from odcd.utils.saving import load_od

EVENT_SOURCE_PREFIX = "seldon.ceserver.odcdserver"
EVENT_TYPE = "seldon.outlier"

HEADER_RETURN_FEATURE_SCORE = "odcd-return-feature-score"
HEADER_RETURN_INSTANCE_SCORE = "odcd-return-instance-score"
HEADER_OUTLIER_TYPE = "odcd-outlier-type"

class ODCDModel(ceserver.CEModel):  # pylint:disable=c-extension-no-member
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
        self.odcd = None

    def load(self):
        """
        Load the model from storage

        """
        model_folder = kfserving.Storage.download(self.storage_uri)
        self.odcd = load_od(model_folder)
        self.ready = True

    def process_event(self, inputs: List, headers: Dict) -> Dict:
        """
        Process the event and return ODCD score

        Parameters
        ----------
        inputs
             Input data
        headers
             Header options

        Returns
        -------
             ODCD response

        """
        try:
            X = np.array(inputs)
        except Exception as e:
            raise Exception(
                "Failed to initialize NumPy array from inputs: %s, %s" % (e, inputs))
        outlier_type = 'instance'

        if HEADER_OUTLIER_TYPE in headers and headers[HEADER_OUTLIER_TYPE]:
            outlier_type = headers[HEADER_OUTLIER_TYPE]
        ret_feature_score = False
        if HEADER_RETURN_FEATURE_SCORE in headers and headers[HEADER_RETURN_FEATURE_SCORE] == "true":
            ret_feature_score = True
        ret_instance_score = False
        if HEADER_RETURN_INSTANCE_SCORE in headers and headers[HEADER_RETURN_INSTANCE_SCORE] == "true":
            ret_instance_score = True

        od_preds = self.odcd.predict(X,
                                     outlier_type=outlier_type,  # use 'feature' or 'instance' level
                                     return_feature_score=ret_feature_score,
                                     # scores used to determine outliers
                                     return_instance_score=ret_instance_score)
        print(od_preds)
        return json.loads(json.dumps(od_preds, cls=NumpyEncoder))

    def event_source(self) -> str:
        return EVENT_SOURCE_PREFIX + self.name

    def event_type(self) -> str:
        return EVENT_TYPE

    def headers(self) -> List:
        return [HEADER_OUTLIER_TYPE, HEADER_RETURN_FEATURE_SCORE, HEADER_RETURN_INSTANCE_SCORE]