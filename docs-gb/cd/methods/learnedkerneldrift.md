---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

[source](../../api/alibi_detect.cd.learned_kernel.rst)

# Learned Kernel

## Overview

The learned-kernel drift detector ([Liu et al., 2020](https://arxiv.org/abs/2002.09116)) is an extension of the [Maximum Mean Discrepancy](./mmddrift.ipynb) drift detector where the kernel used to define the MMD is trained using a portion of the data to maximise an estimate of the resulting test power. Once the kernel has been learned a permutation test is performed in the usual way on the value of the MMD.

This method is closely related to the [classifier drift detector](./classifierdrift.ipynb) which trains a classifier to discriminate between instances from the reference window and instances from the test window. The difference here is that we train a kernel to output high similarity on instances from the same window and low similarity between instances from different windows. If this is possible in a generalisable manner then drift must have occured.

As with the classifier-based approach, we should specify the proportion of data to use for training and testing respectively as well as training arguments such as the learning rate and batch size. Note that a new kernel is trained for each test set that is passed for detection.

## Usage

### Initialize

Arguments:

* `x_ref`: Data used as reference distribution.

* `kernel`: A differentiable **TensorFlow** or **PyTorch** module that takes two sets of instances as inputs and returns a kernel similarity matrix as output.

Keyword arguments:

* `backend`: **TensorFlow**, **PyTorch** and [**KeOps**](https://github.com/getkeops/keops) implementations of the learned kernel detector are available. The backend can be specified as *tensorflow*, *pytorch* or *keops*. Defaults to *tensorflow*.

* `p_val`: p-value threshold used for the significance of the test.

* `preprocess_at_init`: Whether to already apply the (optional) preprocessing step to the reference data at initialization and store the preprocessed data. Dependent on the preprocessing step, this can reduce the computation time for the predict step significantly, especially when the reference dataset is large. Defaults to *True*. It is possible that it needs to be set to *False* if the preprocessing step requires statistics from both the reference and test data, such as the mean or standard deviation.

* `x_ref_preprocessed`: Whether or not the reference data `x_ref` has already been preprocessed. If *True*, the reference data will be skipped and preprocessing will only be applied to the test data passed to `predict`.

* `update_x_ref`: Reference data can optionally be updated to the last N instances seen by the detector or via [reservoir sampling](https://en.wikipedia.org/wiki/Reservoir_sampling) with size N. For the former, the parameter equals *{'last': N}* while for reservoir sampling *{'reservoir_sampling': N}* is passed. If the input data type is of type `List[Any]` then `update_x_ref` needs to be set to *None* and the reference set remains fixed.

* `preprocess_fn`: Function to preprocess the data before computing the data drift metrics.

* `n_permutations`: The number of permutations to use in the permutation test once the MMD has been computed.

* `var_reg`: Constant added to the estimated variance of the MMD for stability.

* `reg_loss_fn`: The regularisation term *reg_loss_fn(kernel)* is added to the loss function being optimized.

* `train_size`: Optional fraction (float between 0 and 1) of the dataset used to train the classifier. The drift is detected on *1 - train_size*.

* `retrain_from_scratch`: Whether the kernel should be retrained from scratch for each set of test data or whether it should instead continue training from where it left off on the previous set. Defaults to *True*.

* `optimizer`: Optimizer used during training of the kernel. From `torch.optim` for PyTorch and `tf.keras.optimizers` for TensorFlow.

* `learning_rate`: Learning rate for the optimizer.

* `batch_size`: Batch size used during training of the kernel.

* `batch_size_predict`: Batch size used for the trained drift detector predictions.

* `preprocess_batch_fn`: Optional batch preprocessing function. For example to convert a list of generic objects to a tensor which can be processed by the kernel.

* `epochs`: Number of training epochs for the kernel.

* `verbose`: Verbosity level during the training of the kernel. 0 is silent and 1 prints a progress bar.

* `train_kwargs`: Optional additional kwargs for the built-in TensorFlow (`from alibi_detect.models.tensorflow import trainer`) or PyTorch (`from alibi_detect.models.pytorch import trainer`) trainer functions.

* `dataset`: Dataset object used during training of the kernel. Defaults to `alibi_detect.utils.pytorch.TorchDataset` (an instance of `torch.utils.data.Dataset`) for the PyTorch and KeOps backends and `alibi_detect.utils.tensorflow.TFDataset` (an instance of `tf.keras.utils.Sequence`) for the TensorFlow backend. For PyTorch or KeOps, the dataset should only take the windows x_ref and x_test as input, so when e.g. *TorchDataset* is passed to the detector at initialisation, during training *TorchDataset(x_ref, x_test)* is used. For TensorFlow, the dataset is an instance of `tf.keras.utils.Sequence`, so when e.g. *TFDataset* is passed to the detector at initialisation, during training *TFDataset(x_ref, x_test, batch_size=batch_size, shuffle=True)* is used. x_ref and x_test can be of type np.ndarray or List[Any].

* `input_shape`: Shape of input data.

* `data_type`: Optionally specify the data type (e.g. tabular, image or time-series). Added to metadata.


Additional PyTorch and KeOps keyword arguments:

* `device`: *cuda* or *gpu* to use the GPU and *cpu* for the CPU. If the device is not specified, the detector will try to leverage the GPU if possible and otherwise fall back on CPU.

* `dataloader`: Dataloader object used during training of the kernel. Defaults to `torch.utils.data.DataLoader`. The dataloader is not initialized yet, this is done during init off the detector using the `batch_size`. Custom dataloaders can be passed as well, e.g. for graph data we can use `torch_geometric.data.DataLoader`.

* `num_workers`: The number of workers used by the `DataLoader`. The default (`num_workers=0`) means multi-process data loading is disabled. Setting `num_workers>0` may be unreliable on Windows.


Additional KeOps only keyword arguments:

* `batch_size_permutations`: KeOps computes the `n_permutations` of the MMD^2 statistics in chunks of `batch_size_permutations`. Defaults to 1,000,000.

### Defining the kernel

Any differentiable *Pytorch* or *TensorFlow* module that takes as input two instances and outputs a scalar (representing similarity) can be used as the kernel for this drift detector. However, in order to ensure that MMD=0 implies no-drift the kernel should satify a *characteristic* property. This can be guaranteed by defining a kernel as $$k(x,y)=(1-\epsilon)*k_a(\Phi(x), \Phi(y)) + \epsilon*k_b(x,y),$$ where $\Phi$ is a learnable projection, $k_a$ and $k_b$ are simple characteristic kernels (such as a [Gaussian RBF](https://en.wikipedia.org/wiki/Radial_basis_function_kernel)), and $\epsilon>0$ is a small constant. By letting $\Phi$ be very flexible we can learn powerful kernels in this manner.

This is easily implemented using the `DeepKernel` class provided in `alibi_detect`. We demonstrate below how we might define a convolutional kernel for images using *Pytorch*. By default `GaussianRBF` kernels are used for $k_a$ and $k_b$ and here we specify $\epsilon=0.01$, but we could alternatively set `eps='trainable'`.

```python
from torch import nn
from alibi_detect.utils.pytorch import DeepKernel

# define the projection phi
proj = nn.Sequential(
    nn.Conv2d(3, 8, 4, stride=2, padding=0),
    nn.ReLU(),
    nn.Conv2d(8, 16, 4, stride=2, padding=0),
    nn.ReLU(),
    nn.Conv2d(16, 32, 4, stride=2, padding=0),
    nn.ReLU(),
    nn.Flatten(),
)

# define the kernel
kernel = DeepKernel(proj, eps=0.01)
```

It is important to note that, if `retrain_from_scratch=True` and we have not initialised the kernel bandwidth `sigma` for the default `GaussianRBF` kernel $k_a$ and optionally also for $k_b$, we will initialise `sigma` using a median (*PyTorch* and *TensorFlow*) or mean (*KeOps*) bandwidth heuristic for every detector prediction. For KeOps detectors specifically, this could form a computational bottleneck and should be avoided by already specifying a bandwidth in advance. To do this, we can leverage the library's built-in heuristics:

```python
from alibi_detect.utils.pytorch.kernels import sigma_median, GaussianRBF

# example usage
x, y = torch.randn(*shape), torch.randn(*shape)
dist = ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)  # distance used for the GaussianRBF kernel
sigma = sigma_median(x, y, dist)
kernel_b = GaussianRBF(sigma=sigma, trainable=True)

# equivalent TensorFlow and KeOps functions
from alibi_detect.utils.tensorflow.kernels import sigma_median
from alibi_detect.utils.keops.kernels import sigma_mean
```

### Instantiating the detector

Instantiating the detector is then as simple as passing the reference data and the kernel as follows:
```python
# instantiate the detector
from alibi_detect.cd import LearnedKernelDrift

cd = LearnedKernelDrift(x_ref, kernel, backend='pytorch', p_val=.05, epochs=10, batch_size=32)
```

We could have alternatively defined the kernel and instantiated the detector using *KeOps*:

```python
from alibi_detect.utils.keops import DeepKernel

kernel = DeepKernel(proj, eps=0.01)
cd = LearnedKernelDrift(x_ref, kernel, backend='keops', p_val=.05, epochs=10, batch_size=32)
```

Or by using *TensorFlow* as the backend:

```python
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Input
from alibi_detect.utils.tensorflow import DeepKernel

# define the projection phi
proj = tf.keras.Sequential(
  [
      Input(shape=(32, 32, 3)),
      Conv2D(8, 4, strides=2, padding='same', activation=tf.nn.relu),
      Conv2D(16, 4, strides=2, padding='same', activation=tf.nn.relu),
      Conv2D(32, 4, strides=2, padding='same', activation=tf.nn.relu),
      Flatten(),
  ]
)

# define the kernel
kernel = DeepKernel(proj, eps=0.01)

# instantiate the detector
cd = LearnedKernelDrift(x_ref, kernel, backend='tensorflow', p_val=.05, epochs=10, batch_size=32)
```

### Detect Drift

We detect data drift by simply calling `predict` on a batch of instances `x`. `return_p_val` equal to *True* will also return the p-value of the test, `return_distance` equal to *True* will return a notion of strength of the drift and `return_kernel` equals *True* will also return the trained kernel.

The prediction takes the form of a dictionary with `meta` and `data` keys. `meta` contains the detector's metadata while `data` is also a dictionary which contains the actual predictions stored in the following keys:

* `is_drift`: 1 if the sample tested has drifted from the reference data and 0 otherwise.

* `threshold`: the user-defined p-value threshold defining the significance of the test

* `p_val`: the p-value of the test if `return_p_val` equals *True*.

* `distance`: MMD^2 metric between the reference data and the new batch if `return_distance` equals *True*.

* `distance_threshold`: MMD^2 metric value from the permutation test which corresponds to the the p-value threshold if `return_distance` equals *True*.

* `kernel`: The trained kernel if `return_kernel` equals *True*.


```python
preds = cd.predict(X, return_p_val=True, return_distance=True)
```

## Examples

### Graph

[Drift detection on molecular graphs](../../examples/cd_mol.ipynb)

### Image

[Drift detection on CIFAR10](../../examples/cd_clf_cifar10.ipynb)

### Tabular

[Scaling up drift detection with KeOps](../../examples/cd_mmd_keops.ipynb)
