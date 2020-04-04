import argparse

import ceserver
import tensorflow as tf
from enum import Enum

tf.keras.backend.clear_session()
import logging
from .od_model import AlibiDetectOutlierModel
from .ad_model import AlibiDetectAdversarialDetectionModel


class AlibiDetectMethod(Enum):
    adversarial_detector = "AdversarialDetector"
    outlier_detector = "OutlierDetector"
    concept_drift = "ConceptDrift"

    def __str__(self):
        return self.value


class GroupedAction(argparse.Action):  # pylint:disable=too-few-public-methods
    def __call__(self, theparser, namespace, values, option_string=None):
        group, dest = self.dest.split(".", 2)
        groupspace = getattr(namespace, group, argparse.Namespace())
        setattr(groupspace, dest, values)
        setattr(namespace, group, groupspace)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


DEFAULT_MODEL_NAME = "model"

parser = argparse.ArgumentParser(parents=[ceserver.server.parser])
parser.add_argument(
    "--model_name",
    default=DEFAULT_MODEL_NAME,
    help="The name that the model is served under.",
)
parser.add_argument("--storage_uri", required=True, help="A URI pointer to the model")

subparsers = parser.add_subparsers(help="sub-command help", dest="command")

# Anchor Tabular Arguments
parser_drift = subparsers.add_parser(str(AlibiDetectMethod.concept_drift))
parser_drift.add_argument(
    "--drift_batch_size",
    type=int,
    action=GroupedAction,
    dest="alibi.drift_batch_size",
    default=argparse.SUPPRESS,
)

parser_adversarial = subparsers.add_parser(str(AlibiDetectMethod.adversarial_detector))

parser_outlier = subparsers.add_parser(str(AlibiDetectMethod.outlier_detector))

args, _ = parser.parse_known_args()

argdDict = vars(args).copy()
if "explainer" in argdDict:
    extra = vars(args.explainer)
else:
    extra = {}
logging.info("Extra args: %s", extra)

if __name__ == "__main__":
    method = AlibiDetectMethod(args.command)
    if method == AlibiDetectMethod.outlier_detector:
        model = AlibiDetectOutlierModel(args.model_name, args.storage_uri)
    elif method == AlibiDetectMethod.adversarial_detector:
        model = AlibiDetectAdversarialDetectionModel(args.model_name, args.storage_uri)
    else:
        logging.error("Unknown method %s", args.command)
    ceserver.CEServer().start(model)
