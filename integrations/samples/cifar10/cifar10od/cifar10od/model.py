from typing import List

import numpy as np
import odcdserver

EVENT_SOURCE_PREFIX = "seldon.ceserver.odcdserver.cifar10."
EVENT_TYPE = "seldon.outlier"


class Cifar10ODCDModel(odcdserver.ODCDModel):  # pylint:disable=c-extension-no-member
    def __init__(self, name: str, model_dir: str):
        super().__init__(name, model_dir)

    def transform(self, inputs: List) -> List:
        X = np.array(inputs)
        X = X / 2.0 + 0.5
        X = np.transpose(X, (0, 2, 3, 1))
        return X.tolist()

    def event_source(self) -> str:
        return EVENT_SOURCE_PREFIX + self.name

    def event_type(self) -> str:
        return EVENT_TYPE
