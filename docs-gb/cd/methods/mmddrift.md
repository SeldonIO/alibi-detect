---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

[source](../../api/alibi_detect.cd.mmd.rst)

# Maximum Mean Discrepancy

## Overview

The [Maximum Mean Discrepancy (MMD)](http://jmlr.csail.mit.edu/papers/v13/gretton12a.html) detector is a kernel-based method for multivariate 2 sample testing. The MMD is a distance-based measure between 2 distributions *p* and *q* based on the mean embeddings $\mu_{p}$ and $\mu_{q}$ in a reproducing kernel Hilbert space $F$:

$$
MMD(F, p, q) = || \mu_{p} - \mu_{q} ||^2_{F}
$$

We can compute unbiased estimates of $MMD^2$ from the samples of the 2 distributions after applying the kernel trick. We use by default a [radial basis function kernel](https://en.wikipedia.org/wiki/Radial_basis_function_kernel), but users are free to pass their own kernel of preference to the detector. We obtain a $p$-value via a [permutation test](https://en.wikipedia.org/wiki/Resampling_(statistics)) on the values of $MMD^2$.

For high-dimensional data, we typically want to reduce the dimensionality before computing the permutation test. Following suggestions in [Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift](https://arxiv.org/abs/1810.11953), we incorporate Untrained AutoEncoders (UAE) and black-box shift detection using the classifier's softmax outputs ([BBSDs](https://arxiv.org/abs/1802.03916)) as out-of-the box preprocessing methods and note that [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) can also be easily implemented using `scikit-learn`. Preprocessing methods which do not rely on the classifier will usually pick up drift in the input data, while BBSDs focuses on label shift.

Detecting input data drift (covariate shift) $\Delta p(x)$ for text data requires a custom preprocessing step. We can pick up changes in the semantics of the input by extracting (contextual) embeddings and detect drift on those. Strictly speaking we are not detecting $\Delta p(x)$ anymore since the whole training procedure (objective function, training data etc) for the (pre)trained embeddings has an impact on the embeddings we extract. The library contains functionality to leverage pre-trained embeddings from [HuggingFace's transformer package](https://github.com/huggingface/transformers) but also allows you to easily use your own embeddings of choice. Both options are illustrated with examples in the [Text drift detection on IMDB movie reviews](../../examples/cd_text_imdb.ipynb) notebook.

## Usage

### Initialize


Arguments:

* `x_ref`: Data used as reference distribution.


Keyword arguments:

* `backend`: **TensorFlow**, **PyTorch** and [**KeOps**](https://github.com/getkeops/keops) implementations of the MMD detector are available. Specify the backend (*tensorflow*, *pytorch* or *keops*). Defaults to *tensorflow*.

* `p_val`: p-value used for significance of the permutation test.

* `preprocess_at_init`: Whether to already apply the (optional) preprocessing step to the reference data at initialization and store the preprocessed data. Dependent on the preprocessing step, this can reduce the computation time for the predict step significantly, especially when the reference dataset is large. Defaults to *True*. It is possible that it needs to be set to *False* if the preprocessing step requires statistics from both the reference and test data, such as the mean or standard deviation.

* `x_ref_preprocessed`: Whether or not the reference data `x_ref` has already been preprocessed. If *True*, the reference data will be skipped and preprocessing will only be applied to the test data passed to `predict`.

* `update_x_ref`: Reference data can optionally be updated to the last N instances seen by the detector or via [reservoir sampling](https://en.wikipedia.org/wiki/Reservoir_sampling) with size N. For the former, the parameter equals *{'last': N}* while for reservoir sampling *{'reservoir_sampling': N}* is passed.

* `preprocess_fn`: Function to preprocess the data before computing the data drift metrics. Typically a dimensionality reduction technique.

* `kernel`: Kernel used when computing the MMD. Defaults to a Gaussian RBF kernel (`from alibi_detect.utils.pytorch import GaussianRBF`, `from alibi_detect.utils.tensorflow import GaussianRBF` or `from alibi_detect.utils.keops import GaussianRBF` dependent on the backend used). Note that for the KeOps backend, the diagonal entries of the kernel matrices `kernel(x_ref, x_ref)` and `kernel(x_test, x_test)` should be equal to 1. This is compliant with the default Gaussian RBF kernel.

* `sigma`: Optional bandwidth for the kernel as a `np.ndarray`. We can also average over a number of different bandwidths, e.g. `np.array([.5, 1., 1.5])`.

* `configure_kernel_from_x_ref`: If `sigma` is not specified, the detector can infer it via a heuristic and set `sigma` to the median (*TensorFlow* and *PyTorch*) or the mean pairwise distance between 2 samples (*KeOps*) by default. If `configure_kernel_from_x_ref` is *True*, we can already set `sigma` at initialization of the detector by inferring it from `x_ref`, speeding up the prediction step. If set to *False*, `sigma` is computed separately for each test batch at prediction time.

* `n_permutations`: Number of permutations used in the permutation test.

* `input_shape`: Optionally pass the shape of the input data.

* `data_type`: can specify data type added to the metadata. E.g. *'tabular'* or *'image'*.


Additional PyTorch keyword arguments:

* `device`: *cuda* or *gpu* to use the GPU and *cpu* for the CPU. If the device is not specified, the detector will try to leverage the GPU if possible and otherwise fall back on CPU.

Additional KeOps keyword arguments:

* `batch_size_permutations`: KeOps computes the `n_permutations` of the MMD^2 statistics in chunks of `batch_size_permutations`. Defaults to 1,000,000.

Initialized drift detector examples for each of the available backends:


```python
from alibi_detect.cd import MMDDrift

cd_tf = MMDDrift(x_ref, backend='tensorflow', p_val=.05)
cd_torch = MMDDrift(x_ref, backend='pytorch', p_val=.05)
cd_keops = MMDDrift(x_ref, backend='keops', p_val=.05)
```

We can also easily add preprocessing functions for the *TensorFlow* and *PyTorch* frameworks. Note that we can also combine for instance a PyTorch preprocessing step with a KeOps detector. The following example uses a randomly initialized image encoder in PyTorch:

```python
from functools import partial
import torch
import torch.nn as nn
from alibi_detect.cd.pytorch import preprocess_drift

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define encoder
encoder_net = nn.Sequential(
    nn.Conv2d(3, 64, 4, stride=2, padding=0),
    nn.ReLU(),
    nn.Conv2d(64, 128, 4, stride=2, padding=0),
    nn.ReLU(),
    nn.Conv2d(128, 512, 4, stride=2, padding=0),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(2048, 32)
).to(device).eval()

# define preprocessing function
preprocess_fn = partial(preprocess_drift, model=encoder_net, device=device, batch_size=512)

cd = MMDDrift(x_ref, backend='pytorch', p_val=.05, preprocess_fn=preprocess_fn)
```
The same functionality is supported in TensorFlow and the main difference is that you would import from `alibi_detect.cd.tensorflow import preprocess_drift`. Other preprocessing steps such as the output of hidden layers of a model or extracted text embeddings using transformer models can be used in a similar way in both frameworks. TensorFlow example for the hidden layer output:

```python
from alibi_detect.cd.tensorflow import HiddenOutput, preprocess_drift

model = # TensorFlow model; tf.keras.Model or tf.keras.Sequential
preprocess_fn = partial(preprocess_drift, model=HiddenOutput(model, layer=-1), batch_size=128)

cd = MMDDrift(x_ref, backend='tensorflow', p_val=.05, preprocess_fn=preprocess_fn)
```

Check out the [Drift detection on CIFAR10](../../examples/cd_mmd_cifar10.ipynb) example for more details.

Alibi Detect also includes custom text preprocessing steps in both TensorFlow and PyTorch based on Huggingface's [transformers](https://github.com/huggingface/transformers) package:

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from alibi_detect.cd.pytorch import preprocess_drift
from alibi_detect.models.pytorch import TransformerEmbedding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

embedding_type = 'hidden_state'
layers = [5, 6, 7]
embed = TransformerEmbedding(model_name, embedding_type, layers)
model = nn.Sequential(embed, nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, enc_dim)).to(device).eval()
preprocess_fn = partial(preprocess_drift, model=model, tokenizer=tokenizer, max_len=512, batch_size=32)

# initialise drift detector
cd = MMDDrift(x_ref, backend='pytorch', p_val=.05, preprocess_fn=preprocess_fn)
```

Again the same functionality is supported in TensorFlow but with `from alibi_detect.cd.tensorflow import preprocess_drift` and `from alibi_detect.models.tensorflow import TransformerEmbedding` imports. Check out the [Text drift detection on IMDB movie reviews](../../examples/cd_text_imdb.ipynb) example for more information.

### Detect Drift

We detect data drift by simply calling `predict` on a batch of instances `x`. We can return the p-value and the threshold of the permutation test by setting `return_p_val` to *True* and the maximum mean discrepancy metric and threshold by setting `return_distance` to *True*.

The prediction takes the form of a dictionary with `meta` and `data` keys. `meta` contains the detector's metadata while `data` is also a dictionary which contains the actual predictions stored in the following keys:

* `is_drift`: 1 if the sample tested has drifted from the reference data and 0 otherwise.

* `p_val`: contains the p-value if `return_p_val` equals *True*.

* `threshold`: p-value threshold if `return_p_val` equals *True*.

* `distance`: MMD^2 metric between the reference data and the new batch if `return_distance` equals *True*.

* `distance_threshold`: MMD^2 metric value from the permutation test which corresponds to the the p-value threshold.


```python
preds = cd.predict(X, return_p_val=True, return_distance=True)
```

## Examples

### Graph

[Drift detection on molecular graphs](../../examples/cd_mol.ipynb)

### Image

[Drift detection on CIFAR10](../../examples/cd_mmd_cifar10.ipynb)

### Tabular

[Scaling up drift detection with KeOps](../../examples/cd_mmd_keops.ipynb)

### Text

[Text drift detection on IMDB movie reviews](../../examples/cd_text_imdb.ipynb)

