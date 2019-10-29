import argparse

import ceserver
import tensorflow as tf

tf.keras.backend.clear_session()

from cifar10od import Cifar10ODCDModel

DEFAULT_MODEL_NAME = "model"

parser = argparse.ArgumentParser(parents=[ceserver.server.parser])
parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME,
                    help='The name that the model is served under.')
parser.add_argument('--model_dir', required=True,
                    help='A URI pointer to the model directory')
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = Cifar10ODCDModel(args.model_name, args.model_dir)
    ceserver.CEServer().start(model)
