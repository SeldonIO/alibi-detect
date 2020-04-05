import json
from typing import List, Dict, Optional
import logging
import ceserver
import kfserving
import numpy as np
from .numpy_encoder import NumpyEncoder
from alibi_detect.utils.saving import load_detector, Data
from adserver.constants import HEADER_RETURN_INSTANCE_SCORE, HEADER_RETURN_FEATURE_SCORE, \
    HEADER_OUTLIER_TYPE


class AlibiDetectModel(ceserver.CEModel):  # pylint:disable=c-extension-no-member
    def __init__(self, name: str, storage_uri: str, model:Optional[Data] = None):
        """
        Outlier Detection Model

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
        self.model: Optional[Data] = model

    def load(self):
        """
        Load the model from storage

        """
        model_folder = kfserving.Storage.download(self.storage_uri)
        self.model: Data = load_detector(model_folder)
        self.ready = True

