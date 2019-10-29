from typing import List, Dict

DEFAULT_EVENT_PREFIX = "seldon.ceserver."


class CEModel(object):

    def __init__(self, name: str):
        self.name = name
        self.ready = False

    def load(self):
        raise NotImplementedError

    def transform(self, inputs: List) -> List:
        return inputs

    def process_event(self, inputs: List, headers: Dict) -> Dict:
        raise NotImplementedError

    def event_source(self) -> str:
        return DEFAULT_EVENT_PREFIX + self.name

    def event_type(self) -> str:
        return DEFAULT_EVENT_PREFIX + self.name

    def headers(self) -> List:
        return []