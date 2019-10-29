from typing import List, Dict

DEFAULT_EVENT_PREFIX = "seldon.ceserver."


class CEModel(object):

    def __init__(self, name: str):
        """
        A CloudEvents model

        Parameters
        ----------
        name
             The name of the model
        """
        self.name = name
        self.ready = False

    def load(self):
        """
        Load the model

        """
        raise NotImplementedError

    def transform(self, inputs: List) -> List:
        """
        Transformation

        Parameters
        ----------
        inputs
             Input data

        Returns
        -------
             Transformed data

        """
        return inputs

    def process_event(self, inputs: List, headers: Dict) -> Dict:
        """
        Process the event data and return a response

        Parameters
        ----------
        inputs
             Input data
        headers
             Headers from the request

        Returns
        -------
             A response object

        """
        raise NotImplementedError

    def event_source(self) -> str:
        """
        Returns
        -------
             The event source name

        """
        return DEFAULT_EVENT_PREFIX + self.name

    def event_type(self) -> str:
        """

        Returns
        -------
             The event type

        """
        return DEFAULT_EVENT_PREFIX + self.name

    def headers(self) -> List:
        """

        Returns
        -------
             A desired list of header keys to extract from request

        """
        return []