import argparse

import ceserver
import tensorflow as tf

tf.keras.backend.clear_session()

from .model import Cifar10OutlierModel

DEFAULT_MODEL_NAME = "model"

parser = argparse.ArgumentParser(parents=[ceserver.server.parser])
parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME,
                    help='The name that the model is served under.')
parser.add_argument('--storage_uri', required=True,
                    help='A URI pointer to the model')
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = Cifar10OutlierModel(args.model_name, args.storage_uri)
    ceserver.CEServer().start(model)
