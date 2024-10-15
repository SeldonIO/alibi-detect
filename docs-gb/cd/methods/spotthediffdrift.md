---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

[source](../../api/alibi_detect.cd.spot_the_diff.rst)

# Spot-the-diff

## Overview

The spot-the-diff drift detector is an extension of the [Classifier](./classifierdrift.ipynb) drift detector where the classifier is specified in a manner that makes detections interpretable at the feature level when they occur. The detector is inspired by the work of [Jitkrittum et al. (2016)](https://arxiv.org/abs/1605.06796) but various major adaptations have been made.

As with the usual classifier-based approach, a portion of the available data is used to train a classifier that can disciminate reference instances from test instances. If the classifier can learn to discriminate in a generalisable manner then drift must have occured. Here we additionally enforce that the classifier takes the form $$\text{logit}(\hat{p}_T) = b_0 + b_1 k(x,w_1) + ... + b_Jk(x,w_J),$$ where $\hat{p}_T$ is the predicted probability that instance $x$ is from the test window (rather than reference), $k(\cdot,\cdot)$ is a kernel specifying a notion of similarity between instances, $w_i$ are learnable *test locations* and $b_i$ are learnable regression coefficients.

If the detector flags drift and $b_i >0$ then we know that it reached its decision by considering how similar each instance is to the instance $w_i$, with those being more similar being more likely to be test instances than reference instances. Alternatively if $b_i < 0$ then instances more similar to $w_i$ were deemed more likely to be reference instances.

In order to provide less noisy and therefore more interpretable results, we define each test location as $$w_i = \bar{x}+d_i$$ where $\bar{x}$ is the mean reference instance. We may then interpret $d_i$ as the additive transformation deemed to make the average reference more ($b_i>0$) or less ($b_i<0$) similar to a test instance. Defining the test locations in this way allows us to instead learn the *difference* $d_i$ and apply regularisation such that non-zero values must be justified by improved classification performance.  This allows us to more clearly identify which features any detected drift should be attributed to.

As with the standard classifier-based approach, we should specify the proportion of data to use for training and testing respectively as well as training arguments such as the learning rate and batch size. Note that classifier is trained for each test set that is passed for detection.

## Usage

### Initialize

Arguments:

* `x_ref`: Data used as reference distribution.

Keyword arguments:

* `backend`: Specify the backend (*tensorflow* or *pytorch*) to use for defining the kernel and training the test locations/differences.

* `p_val`: p-value threshold used for the significance of the test.

* `preprocess_fn`: Function to preprocess the data before computing the data drift metrics.

* `kernel`: A differentiable **TensorFlow** or **PyTorch** module that takes two instances as input and returns a scalar notion of similarity os output. Defaults to a Gaussian radial basis function.

* `n_diffs`:  The number of test locations to use, each corresponding to an interpretable difference.

* `initial_diffs`: Array used to initialise the diffs that will be learned. Defaults to Gaussian for each feature with equal variance to that of reference data.

* `l1_reg`: Strength of l1 regularisation to apply to the differences.

* `binarize_preds`: Whether to test for discrepency on soft  (e.g. probs/logits) model predictions directly with a K-S test or binarise to 0-1 prediction errors and apply a binomial test.

* `train_size`: Optional fraction (float between 0 and 1) of the dataset used to train the classifier. The drift is detected on *1 - train_size*. Cannot be used in combination with `n_folds`.

* `n_folds`: Optional number of stratified folds used for training. The model preds are then calculated on all the out-of-fold instances. This allows to leverage all the reference and test data for drift detection at the expense of longer computation. If both `train_size` and `n_folds` are specified, `n_folds` is prioritized.

* `retrain_from_scratch`: Whether the classifier should be retrained from scratch for each set of test data or whether it should instead continue training from where it left off on the previous set.

* `seed`: Optional random seed for fold selection.

* `optimizer`: Optimizer used during training of the kernel. From `torch.optim` for PyTorch and `tf.keras.optimizers` for TensorFlow.

* `learning_rate`: Learning rate for the optimizer.

* `batch_size`: Batch size used during training of the kernel.

* `preprocess_batch_fn`: Optional batch preprocessing function. For example to convert a list of generic objects to a tensor which can be processed by the kernel.

* `epochs`: Number of training epochs for the kernel.

* `verbose`: Verbosity level during the training of the kernel. 0 is silent and 1 prints a progress bar.

* `train_kwargs`: Optional additional kwargs for the built-in TensorFlow (`from alibi_detect.models.tensorflow import trainer`) or PyTorch (`from alibi_detect.models.pytorch import trainer`) trainer functions.

* `dataset`: Dataset object used during training of the classifier. Defaults to `alibi_detect.utils.pytorch.TorchDataset` (an instance of `torch.utils.data.Dataset`) for the PyTorch backend and `alibi_detect.utils.tensorflow.TFDataset` (an instance of `tf.keras.utils.Sequence`) for the TensorFlow backend. For PyTorch, the dataset should only take the data x and the array of labels y as input, so when e.g. *TorchDataset* is passed to the detector at initialisation, during training *TorchDataset(x, y)* is used. For TensorFlow, the dataset is an instance of `tf.keras.utils.Sequence`, so when e.g. *TFDataset* is passed to the detector at initialisation, during training *TFDataset(x, y, batch_size=batch_size, shuffle=True)* is used. x can be of type np.ndarray or List[Any] while y is of type np.ndarray.

* `input_shape`: Shape of input data.

* `data_type`: Optionally specify the data type (e.g. tabular, image or time-series). Added to metadata.


Additional PyTorch keyword arguments:

* `device`: *cuda* or *gpu* to use the GPU and *cpu* for the CPU. If the device is not specified, the detector will try to leverage the GPU if possible and otherwise fall back on CPU.

* `dataloader`: Dataloader object used during training of the classifier. Defaults to `torch.utils.data.DataLoader`. The dataloader is not initialized yet, this is done during init off the detector using the `batch_size`. Custom dataloaders can be passed as well, e.g. for graph data we can use `torch_geometric.data.DataLoader`.


### Defining the kernel

Any differentiable *Pytorch* or *TensorFlow* module that takes as input two instances and outputs a scalar (representing similarity) can be used as the kernel for this drift detector. By default a simple [Gaussian RBF](https://en.wikipedia.org/wiki/Radial_basis_function_kernel) kernel is used. Keeping the kernel simple can aid interpretability, but alternatively a "deep kernel" of the form $$k(x,y)=(1-\epsilon)*k_a(\Phi(x), \Phi(y)) + \epsilon*k_b(x,y),$$ where $\Phi$ is a (differentiable) projection, $k_a$ and $k_b$ are simple kernels (such as a Gaussian RBF) and $\epsilon>0$ a small constant can be used. The `DeepKernel` class found in either `alibi_detect.utils.tensorflow` or `alibi_detect.utils.pytorch` aims to make defining such kernels straightforward. You should not allow too many learnable parameters however as we would like the classifier to discriminate using the test locations rather than kernel parameters.


### Instantiating the detector

Instantiating the detector is as simple as passing your reference data and selecting a backend, but you should also consider the number of "diffs" you would like your model to use to discriminate reference from test instances and the strength of regularisation you would like to apply to them.

Using `n_diffs=1` is the simplest to interpret and seems to work well in practice. Using more diffs may result in stronger detection power but the diffs may be harder to interpret due to intereactions and conditional dependencies.

The strength of the regularisation (`l1_reg`) to apply to the diffs should also be specified. Stronger regularisation results in sparser diffs as the classifier is encouraged to discriminate using fewer features. This may make the diff more interpretable but may again come at the cost of detection power.

```python
from alibi_detect.cd import SpotTheDiffDrift

cd = SpotTheDiffDrift(
    x_ref,
    backend='pytorch',
    p_val=.05,
    n_diffs=1,
    l1_reg=1e-3,
    epochs=10,
    batch_size=32
)

```

Alternatively we could have used the *TensorFlow* backend and defined a deep kernel with a convolutional structure:

```python
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Input
from alibi_detect.utils.tensorflow import DeepKernel

# define the projection phi with not too much flexability
proj = tf.keras.Sequential(
  [
      Input(shape=(32, 32, 3)),
      Conv2D(8, 4, strides=2, padding='same', activation=tf.nn.relu),
      Conv2D(16, 4, strides=2, padding='same', activation=tf.nn.relu, trainable=False),
      Conv2D(32, 4, strides=2, padding='same', activation=tf.nn.relu, trainable=False),
      Flatten(),
  ]
)

# define the kernel
kernel = DeepKernel(proj, eps=0.01)

# instantiate the detector
cd = SpotTheDiffDrift(
    x_ref,
    backend='tensorflow',
    p_val=.05,
    kernel=kernel,
    n_diffs=1,
    l1_reg=1e-3,
    epochs=10,
    batch_size=32
)
```

### Detect Drift

We detect data drift by simply calling `predict` on a batch of instances `x`. `return_p_val` equal to *True* will also return the p-value of the test, `return_distance` equal to *True* will return a notion of strength of the drift, `return_probs` equals *True* returns the out-of-fold classifier model prediction probabilities on the reference and test data (0 = reference data, 1 = test data) as well as the associated out-of-fold reference and test instances, and `return_kernel` equals *True* will also return the trained kernel.

The prediction takes the form of a dictionary with `meta` and `data` keys. `meta` contains the detector's metadata while `data` is also a dictionary which contains the actual predictions stored in the following keys:

* `is_drift`: 1 if the sample tested has drifted from the reference data and 0 otherwise.

* `diffs`: a numpy array containing the diffs used to discriminate reference from test instances.

* `diff_coeffs` a coefficient correspond to each diff where a coeffient greater than zero implies that the corresponding diff makes the average reference instances *more* similar to a test instance on average and less than zero implies *less* similar.

* `threshold`: the user-defined p-value threshold defining the significance of the test

* `p_val`: the p-value of the test if `return_p_val` equals *True*.

* `distance`: a notion of strength of the drift if `return_distance` equals *True*. Equal to the K-S test statistic assuming `binarize_preds` equals *False* or the relative error reduction over the baseline error expected under the null if `binarize_preds` equals *True*.

* `probs_ref`: the instance level prediction probability for the reference data `x_ref` (0 = reference data, 1 = test data) if `return_probs` is *True*.

* `probs_test`: the instance level prediction probability for the test data `x` if `return_probs` is *true*.

* `x_ref_oof`: the instances associated with `probs_ref` if `return_probs` equals *True*.

* `x_test_oof`: the instances associated with `probs_test` if `return_probs` equals *True*.

* `kernel`: The trained kernel if `return_kernel` equals *True*.


```python
preds = cd.predict(X, return_p_val=True, return_distance=True)
```

## Examples

[Interpretable Drift detection on MNIST and the Wine Quality dataset](../../examples/cd_spot_the_diff_mnist_wine.ipynb)

