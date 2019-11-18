from typing import List, Dict

import numpy as np
import adserver
from adserver.model import HEADER_RETURN_FEATURE_SCORE

EVENT_SOURCE_PREFIX = "seldon.ceserver.adserver.cifar10."
EVENT_TYPE = "seldon.outlier"


class Cifar10OutlierModel(adserver.AlibiDetectModel):  # pylint:disable=c-extension-no-member
    def __init__(self, name: str, storage_uri: str):
        """
        CIFAR10 Outlier Model

        Parameters
        ----------
        name
             Name of the model
        storage_uri
             Storage location
        """
        super().__init__(name, storage_uri)

    def transform(self, inputs: List) -> List:
        """
        Transform the request to that expected by the model.

        Parameters
        ----------
        inputs
             Raw inputs

        Returns
        -------
             Transformed inputs

        """
        X = np.array(inputs)
        X = X / 2.0 + 0.5
        X = np.transpose(X, (0, 2, 3, 1))
        return X.tolist()

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
        if not HEADER_RETURN_FEATURE_SCORE in headers:
            headers[HEADER_RETURN_FEATURE_SCORE] = "false"
        return super().process_event(inputs, headers)

    def event_source(self) -> str:
        return EVENT_SOURCE_PREFIX + self.name

    def event_type(self) -> str:
        return EVENT_TYPE
