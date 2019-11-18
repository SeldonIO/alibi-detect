import adserver

EVENT_SOURCE_PREFIX = "seldon.ceserver.adserver.signs."
EVENT_TYPE = "seldon.adversarial"


class SignsAdversarialModel(adserver.AlibiDetectModel):  # pylint:disable=c-extension-no-member
    def __init__(self, name: str, storage_uri: str):
        """
        traffic Signs Adversarial Detection Model

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
