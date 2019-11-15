import kfserving
from typing import List, Dict
import numpy as np
from alibi_detect.utils.saving import load_tf_model
import argparse

class MnistModel(kfserving.KFModel):
    def __init__(self, name: str, model_dir: str):
        super().__init__(name)
        self.model_dir = model_dir
        self.name = name
        self.ready = False

    def load(self):
        self.model = load_tf_model(self.model_dir)
        self.ready = True

    def predict(self, request: Dict) -> Dict:
        inputs = np.array(request["instances"])

        try:
            preds = self.model.predict(inputs)
            print(preds)
            return { "predictions":  preds.argmax(axis=1).tolist() }
        except Exception as e:
            raise Exception("Failed to predict %s" % e)

DEFAULT_MODEL_NAME = "model"
DEFAULT_LOCAL_MODEL_DIR = "/tmp/model"

parser = argparse.ArgumentParser(parents=[kfserving.kfserver.parser])
parser.add_argument('--model_dir', required=True,
                    help='A URI pointer to the model binary')
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = MnistModel("mnist",args.model_dir)
    # Set number of workers to 1 as model is quite large
    kfserving.KFServer(workers=1).start([model])