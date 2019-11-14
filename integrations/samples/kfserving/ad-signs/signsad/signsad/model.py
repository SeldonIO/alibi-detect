import odcdserver

EVENT_SOURCE_PREFIX = "seldon.ceserver.odcdserver.signs."
EVENT_TYPE = "seldon.adversarial"


class SignsODCDModel(odcdserver.ODCDModel):  # pylint:disable=c-extension-no-member
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

    def event_source(self) -> str:
        return EVENT_SOURCE_PREFIX + self.name

    def event_type(self) -> str:
        return EVENT_TYPE
