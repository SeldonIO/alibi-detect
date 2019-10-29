import argparse
import json
import logging
import os
import uuid
from enum import Enum
from http import HTTPStatus
from typing import Dict, Optional

import requests
import tornado.httpserver
import tornado.ioloop
import tornado.web
from ceserver.model import CEModel
from ceserver.protocols.request_handler import RequestHandler
from ceserver.protocols.seldon_http import SeldonRequestHandler
from ceserver.protocols.tensorflow_http import TensorflowRequestHandler
from cloudevents.sdk import converters
from cloudevents.sdk import marshaller
from cloudevents.sdk.event import v02

DEFAULT_HTTP_PORT = 8080
ODCD_REQUEST_HEADER_PREFIX = "odcd-"

class Protocol(Enum):
    tensorflow_http = "tensorflow.http"
    seldon_http = "seldon.http"

    def __str__(self):
        return self.value


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--http_port', default=DEFAULT_HTTP_PORT, type=int,
                    help='The HTTP Port listened to by the model server.')
parser.add_argument('--protocol', type=Protocol, choices=list(Protocol),
                    default="tensorflow.http",
                    help='The protocol served by the model server')
parser.add_argument('--reply_url', type=str, default="", help='URL to send reply cloudevent')
args, _ = parser.parse_known_args()

CESERVER_LOGLEVEL = os.environ.get('CESERVER_LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=CESERVER_LOGLEVEL)


class CEServer(object):
    def __init__(self, protocol: Protocol = args.protocol, http_port: int = args.http_port,
                 reply_url: str = args.reply_url):
        self.registered_model: CEModel = None
        self.http_port = http_port
        self.protocol = protocol
        self.reply_url = reply_url
        self._http_server: Optional[tornado.httpserver.HTTPServer] = None

    def create_application(self):
        return tornado.web.Application([
            # Outlier detector
            (r"/", EventHandler,
             dict(protocol=self.protocol, model=self.registered_model, reply_url=self.reply_url)),
            # Protocol Discovery API that returns the serving protocol supported by this server.
            (r"/protocol", ProtocolHandler, dict(protocol=self.protocol)),
            # Prometheus Metrics API that returns metrics for model servers
            (r"/v1/metrics", MetricsHandler, dict(model=self.registered_model)),
        ])

    def start(self, model: CEModel):
        self.register_model(model)

        self._http_server = tornado.httpserver.HTTPServer(self.create_application())

        logging.info("Listening on port %s" % self.http_port)
        self._http_server.bind(self.http_port)
        self._http_server.start(1)  # Single worker at present
        tornado.ioloop.IOLoop.current().start()

    def register_model(self, model: CEModel):
        if not model.name:
            raise Exception("Failed to register model, model.name must be provided.")
        self.registered_model = model
        logging.info("Registering model:" + model.name)


def get_request_handler(protocol, request: Dict) -> RequestHandler:
    if protocol == Protocol.tensorflow_http:
        return TensorflowRequestHandler(request)
    else:
        return SeldonRequestHandler(request)


# Send Cloudevent to URL
def sendCloudEvent(event: v02.Event, url: str):
    http_marshaller = marshaller.NewDefaultHTTPMarshaller()
    binary_headers, binary_data = http_marshaller.ToRequest(
        event, converters.TypeBinary, json.dumps)

    print("binary CloudEvent")
    for k, v in binary_headers.items():
        print("{0}: {1}\r\n".format(k, v))
    print(binary_data)

    response = requests.post(url,
                             headers=binary_headers,
                             data=binary_data)
    response.raise_for_status()


class EventHandler(tornado.web.RequestHandler):

    def initialize(self, protocol: str, model: CEModel, reply_url: str):
        self.protocol = protocol
        self.model = model
        self.reply_url = reply_url

    def post(self, name: str = ""):
        if not self.model.ready:
            self.model.load()

        try:
            body = json.loads(self.request.body)
        except json.decoder.JSONDecodeError as e:
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason="Unrecognized request format: %s" % e
            )

        # Extract payload from request
        request_handler: RequestHandler = get_request_handler(self.protocol, body)
        request_handler.validate()
        request = request_handler.extract_request()

        # Create event from request body
        event = v02.Event()
        http_marshaller = marshaller.NewDefaultHTTPMarshaller()
        event = http_marshaller.FromRequest(
            event, self.request.headers, self.request.body, json.loads)
        logging.debug(json.dumps(event.Properties()))

        # Extract any desired request headers
        headers = {}
        for header in self.model.headers():
            headers[header] = self.request.headers.get(header)

        transformed = self.model.transform(request)
        response = self.model.process_event(transformed, headers)
        responseStr = json.dumps(response)

        # Create event from response if reply_url is active
        if not self.reply_url == "":
            if event.EventID() is None or event.EventID() == "":
                resp_event_id = uuid.uuid1().hex
            else:
                resp_event_id = event.EventID()
            revent = (
                v02.Event().
                    SetContentType("application/json").
                    SetData(responseStr).
                    SetEventID(resp_event_id).
                    SetSource(self.model.event_source()).
                    SetEventType(self.model.event_type())
            )
            logging.debug(json.dumps(revent.Properties()))
            sendCloudEvent(revent, self.reply_url)

        self.write(json.dumps(response))


class LivenessHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Alive")


class ProtocolHandler(tornado.web.RequestHandler):
    def initialize(self, protocol: Protocol):
        self.protocol = protocol

    def get(self):
        self.write(str(self.protocol.value))


class MetricsHandler(tornado.web.RequestHandler):
    def initialize(self, model: CEModel):
        self.model = model

    def get(self):
        self.write("Not Implemented")
