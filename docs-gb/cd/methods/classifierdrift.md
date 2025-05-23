---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

[source](../../api/alibi_detect.cd.classifier.rst)

# Classifier

## Overview

The classifier-based drift detector [Lopez-Paz and Oquab, 2017](https://openreview.net/forum?id=SJkXfE5xx) simply tries to correctly distinguish instances from the reference set vs. the test set. The classifier is trained to output the probability that a given instance belongs to the test set. If the probabilities it assigns to unseen test instances are significantly higher (as determined by a Kolmogorov-Smirnov test) to those it assigns to unseen reference instances then the test set must differ from the reference set and drift is flagged. Alternatively, the detector also allows to binarize the classifier predictions (0 or 1) and apply a binomial test on the binarized predictions of the reference vs. the test data. To leverage all the available reference and test data, stratified cross-validation can be applied and the out-of-fold predictions are used for the significance test. Note that a new classifier is trained for each test set or even each fold within the test set.

## Usage

### Initialize

Arguments:

* `x_ref`: Data used as reference distribution.

* `model`: Binary classification model used for drift detection. **TensorFlow**, **PyTorch** and **Sklearn** models are supported.


Keyword arguments:

* `backend`: Specify the backend (*tensorflow*, *pytorch* or *sklearn*). This depends on the framework of the `model`. Defaults to *tensorflow*.

* `p_val`: p-value threshold used for the significance of the test.

* `preprocess_at_init`: Whether to already apply the (optional) preprocessing step to the reference data at initialization and store the preprocessed data. Dependent on the preprocessing step, this can reduce the computation time for the predict step significantly, especially when the reference dataset is large. Defaults to *True*. It is possible that it needs to be set to *False* if the preprocessing step requires statistics from both the reference and test data, such as the mean or standard deviation.

* `x_ref_preprocessed`: Whether or not the reference data `x_ref` has already been preprocessed. If *True*, the reference data will be skipped and preprocessing will only be applied to the test data passed to `predict`.

* `update_x_ref`: Reference data can optionally be updated to the last N instances seen by the detector or via [reservoir sampling](https://en.wikipedia.org/wiki/Reservoir_sampling) with size N. For the former, the parameter equals *{'last': N}* while for reservoir sampling *{'reservoir_sampling': N}* is passed. If the input data type is of type `List[Any]` then `update_x_ref` needs to be set to *None* and the reference set remains fixed.

* `preprocess_fn`: Function to preprocess the data before computing the data drift metrics.

* `preds_type`: Whether the model outputs 'probs' (probabilities - for 'tensorflow', 'pytorch', 'sklearn' models), 'logits' (for 'pytorch', 'tensorflow' models), 'scores' (for 'sklearn' models if `decision_function` is supported).

* `binarize_preds`: Whether to test for discrepancy on soft (e.g. probs/logits/scores) model predictions directly with a K-S test or binarise to 0-1 prediction errors and apply a binomial test. Defaults to *False* and therefore applies the K-S test.

* `train_size`: Optional fraction (float between 0 and 1) of the dataset used to train the classifier. The drift is detected on *1 - train_size*. Cannot be used in combination with `n_folds`.

* `n_folds`: Optional number of stratified folds used for training. The model preds are then calculated on all the out-of-fold predictions. This allows to leverage all the reference and test data for drift detection at the expense of longer computation. If both `train_size` and `n_folds` are specified, `n_folds` is prioritized.

* `seed`: Optional random seed for fold selection.

* `optimizer`: Optimizer used during training of the classifier. From `torch.optim` for PyTorch and `tf.keras.optimizers` for TensorFlow.

* `learning_rate`: Learning rate for the optimizer. Only relevant for *tensorflow* and *pytorch* backends.

* `batch_size`: Batch size used during training of the classifier.Only relevant for *tensorflow* and *pytorch* backends.

* `epochs`: Number of training epochs for the classifier. Applies to each fold if `n_folds` is specified. Only relevant for *tensorflow* and *pytorch* backends.

* `verbose`: Verbosity level during the training of the classifier. 0 is silent and 1 prints a progress bar. Only relevant for *tensorflow* and *pytorch* backends.

* `train_kwargs`: Optional additional kwargs for the built-in TensorFlow (`from alibi_detect.models.tensorflow import trainer`) or PyTorch (`from alibi_detect.models.pytorch import trainer`) trainer functions.

* `dataset`: Dataset object used during training of the classifier. Defaults to `alibi_detect.utils.pytorch.TorchDataset` (an instance of `torch.utils.data.Dataset`) for the PyTorch backend and `alibi_detect.utils.tensorflow.TFDataset` (an instance of `tf.keras.utils.Sequence`) for the TensorFlow backend. For PyTorch, the dataset should only take the data x and the array of labels y as input, so when e.g. *TorchDataset* is passed to the detector at initialisation, during training *TorchDataset(x, y)* is used. For TensorFlow, the dataset is an instance of `tf.keras.utils.Sequence`, so when e.g. *TFDataset* is passed to the detector at initialisation, during training *TFDataset(x, y, batch_size=batch_size, shuffle=True)* is used. x can be of type np.ndarray or List[Any] while y is of type np.ndarray.

* `input_shape`: Shape of input data.

* `data_type`: Optionally specify the data type (e.g. tabular, image or time-series). Added to metadata.


Additional PyTorch keyword arguments:

* `device`: *cuda* or *gpu* to use the GPU and *cpu* for the CPU. If the device is not specified, the detector will try to leverage the GPU if possible and otherwise fall back on CPU.

* `dataloader`: Dataloader object used during training of the model. Defaults to `torch.utils.data.DataLoader`. The dataloader is not initialized yet, this is done during init off the detector using the `batch_size`. Custom dataloaders can be passed as well, e.g. for graph data we can use `torch_geometric.data.DataLoader`.

Additional Sklearn keyword arguments:

* `use_calibration` : Whether to use calibration. Calibration can be used on top of any model. Only relevant for 'sklearn' backend.

* `calibration_kwargs` : Optional additional kwargs for calibration. Only relevant for 'sklearn' backend. See https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html for more details.

* `use_oob` : Whether to use out-of-bag(OOB) predictions. Supported only for `RandomForestClassifier`.


Initialized **TensorFlow** drift detector example:

```python
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from alibi_detect.cd import ClassifierDrift

model = tf.keras.Sequential(
  [
      Input(shape=(32, 32, 3)),
      Conv2D(8, 4, strides=2, padding='same', activation=tf.nn.relu),
      Conv2D(16, 4, strides=2, padding='same', activation=tf.nn.relu),
      Conv2D(32, 4, strides=2, padding='same', activation=tf.nn.relu),
      Flatten(),
      Dense(2, activation='softmax')
  ]
)

cd = ClassifierDrift(x_ref, model, p_val=.05, preds_type='probs', n_folds=5, epochs=2)
```


A similar detector using **PyTorch**:

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(3, 8, 4, stride=2, padding=0),
    nn.ReLU(),
    nn.Conv2d(8, 16, 4, stride=2, padding=0),
    nn.ReLU(),
    nn.Conv2d(16, 32, 4, stride=2, padding=0),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(128, 2)
)

cd = ClassifierDrift(x_ref, model, backend='pytorch', p_val=.05, preds_type='logits')
```

### Detect Drift

We detect data drift by simply calling `predict` on a batch of instances `x`. `return_p_val` equal to *True* will also return the p-value of the test, `return_distance` equal to *True* will return a notion of strength of the drift and `return_probs` equals *True* also returns the out-of-fold classifier model prediction probabilities on the reference and test data (0 = reference data, 1 = test data) as well as the associated out-of-fold reference and test instances.

The prediction takes the form of a dictionary with `meta` and `data` keys. `meta` contains the detector's metadata while `data` is also a dictionary which contains the actual predictions stored in the following keys:

* `is_drift`: 1 if the sample tested has drifted from the reference data and 0 otherwise.

* `threshold`: the user-defined threshold defining the significance of the test

* `p_val`: the p-value of the test if `return_p_val` equals *True*.

* `distance`: a notion of strength of the drift if `return_distance` equals *True*. Equal to the K-S test statistic assuming `binarize_preds` equals *False* or the relative error reduction over the baseline error expected under the null if `binarize_preds` equals *True*.

* `probs_ref`: the instance level prediction probability for the reference data `x_ref` (0 = reference data, 1 = test data) if `return_probs` is *True*.

* `probs_test`: the instance level prediction probability for the test data `x` if `return_probs` is *true*.

* `x_ref_oof`: the instances associated with `probs_ref` if `return_probs` equals *True*.

* `x_test_oof`: the instances associated with `probs_test` if `return_probs` equals *True*.

```python
preds = cd.predict(x)
```

## Examples

[Drift detection on CIFAR10](../../examples/cd_clf_cifar10.ipynb)

[Drift detection on Adult Census](../../examples/cd_clf_adult.ipynb)

[Drift detection on Amazon reviews](../../examples/cd_text_amazon.ipynb)

