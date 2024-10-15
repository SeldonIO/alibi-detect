---
jupyter:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

[source](../../api/alibi_detect.cd.mmd_online.rst)

# Online Maximum Mean Discrepancy

## Overview

The online [Maximum Mean Discrepancy (MMD)](http://jmlr.csail.mit.edu/papers/v13/gretton12a.html) detector is a kernel-based method for online drift detection. The MMD is a distance-based measure between 2 distributions *p* and *q* based on the mean embeddings $\mu_{p}$ and $\mu_{q}$ in a reproducing kernel Hilbert space $F$:

$$
MMD(F, p, q) = || \mu_{p} - \mu_{q} ||^2_{F}
$$

Given reference samples $\{X_i\}_{i=1}^{N}$ and test samples $\{Y_i\}_{i=t}^{t+W}$ we may compute an unbiased estimate $\widehat{MMD}^2(F, \{X_i\}_{i=1}^N, \{Y_i\}_{i=t}^{t+W})$ of the squared MMD between the two underlying distributions. The estimate can be updated at low-cost as new data points enter into the test-window. We use by default a [radial basis function kernel](https://en.wikipedia.org/wiki/Radial_basis_function_kernel), but users are free to pass their own kernel of preference to the detector.

Online detectors assume the reference data is large and fixed and operate on single data points at a time (rather than batches). These data points are passed into the test-window and a two-sample test-statistic (in this case squared MMD) between the reference data and test-window is computed at each time-step. When the test-statistic exceeds a preconfigured threshold, drift is detected. Configuration of the thresholds requires specification of the expected run-time (ERT) which specifies how many time-steps that the detector, on average, should run for in the absence of drift before making a false detection. It also requires specification of a test-window size, with smaller windows allowing faster response to severe drift and larger windows allowing more power to detect slight drift.

For high-dimensional data, we typically want to reduce the dimensionality before passing it to the detector. Following suggestions in [Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift](https://arxiv.org/abs/1810.11953), we incorporate Untrained AutoEncoders (UAE) and black-box shift detection using the classifier's softmax outputs ([BBSDs](https://arxiv.org/abs/1802.03916)) as out-of-the box preprocessing methods and note that [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) can also be easily implemented using `scikit-learn`. Preprocessing methods which do not rely on the classifier will usually pick up drift in the input data, while BBSDs focuses on label shift.

Detecting input data drift (covariate shift) $\Delta p(x)$ for text data requires a custom preprocessing step. We can pick up changes in the semantics of the input by extracting (contextual) embeddings and detect drift on those. Strictly speaking we are not detecting $\Delta p(x)$ anymore since the whole training procedure (objective function, training data etc) for the (pre)trained embeddings has an impact on the embeddings we extract. The library contains functionality to leverage pre-trained embeddings from [HuggingFace's transformer package](https://github.com/huggingface/transformers) but also allows you to easily use your own embeddings of choice. Both options are illustrated with examples in the [Text drift detection on IMDB movie reviews](../../examples/cd_text_imdb.ipynb) notebook.

## Usage

### Initialize


Arguments:

* `x_ref`: Data used as reference distribution.
* `ert`: The expected run-time in the absence of drift, starting from *t=0*.
* `window_size`: The size of the sliding test-window used to compute the test-statistic. Smaller windows focus on responding quickly to severe drift, larger windows focus on ability to detect slight drift.


Keyword arguments:

* `backend`: Backend used for the MMD implementation and configuration.
* `preprocess_fn`: Function to preprocess the data before computing the data drift metrics.
* `kernel`: Kernel used for the MMD computation, defaults to Gaussian RBF kernel.
* `sigma`: Optionally set the GaussianRBF kernel bandwidth. Can also pass multiple bandwidth values as an array. The kernel evaluation is then averaged over those bandwidths.  If `sigma` is not specified, the 'median heuristic' is adopted whereby `sigma` is set as the median pairwise distance between reference samples.
* `n_bootstraps`: The number of bootstrap simulations used to configure the thresholds. The larger this is the more accurately the desired ERT will be targeted. Should ideally be at least an order of magnitude larger than the ERT.
* `verbose`: Whether or not to print progress during configuration.
* `input_shape`: Shape of input data.
* `data_type`: Optionally specify the data type (tabular, image or time-series). Added to metadata.


Additional PyTorch keyword arguments:

* `device`: Device type used. The default None tries to use the GPU and falls back on CPU if needed. Can be specified by passing either 'cuda', 'gpu' or 'cpu'. Only relevant for 'pytorch' backend.


Initialized drift detector example:


```python
from alibi_detect.cd import MMDDriftOnline

cd = MMDDriftOnline(x_ref, ert, window_size, backend='tensorflow')
```

The same detector in PyTorch:

```python
cd = MMDDriftOnline(x_ref, ert, window_size, backend='pytorch')
```

We can also easily add preprocessing functions for both frameworks. The following example uses a randomly initialized image encoder in PyTorch:

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

cd = MMDDriftOnline(x_ref, ert, window_size, backend='pytorch', preprocess_fn=preprocess_fn)
```
The same functionality is supported in TensorFlow and the main difference is that you would import from `alibi_detect.cd.tensorflow import preprocess_drift`. Other preprocessing steps such as the output of hidden layers of a model or extracted text embeddings using transformer models can be used in a similar way in both frameworks. TensorFlow example for the hidden layer output:

```python
from alibi_detect.cd.tensorflow import HiddenOutput, preprocess_drift

model = # TensorFlow model; tf.keras.Model or tf.keras.Sequential
preprocess_fn = partial(preprocess_drift, model=HiddenOutput(model, layer=-1), batch_size=128)

cd = MMDDriftOnline(x_ref, ert, window_size, backend='tensorflow', preprocess_fn=preprocess_fn)
```

Check out the [Online Drift Detection on the Wine Quality Dataset](../../examples/cd_online_wine.ipynb) example for more details.

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
cd = MMDDriftOnline(x_ref, ert, window_size, backend='pytorch', preprocess_fn=preprocess_fn)
```

Again the same functionality is supported in TensorFlow but with `from alibi_detect.cd.tensorflow import preprocess_drift` and `from alibi_detect.models.tensorflow import TransformerEmbedding` imports.

### Detect Drift

We detect data drift by sequentially calling `predict` on single instances `x_t` (no batch dimension) as they each arrive. We can return the test-statistic and the threshold by setting `return_test_stat` to *True*.

The prediction takes the form of a dictionary with `meta` and `data` keys. `meta` contains the detector's metadata while `data` is also a dictionary which contains the actual predictions stored in the following keys:

* `is_drift`: 1 if the test-window (of the most recent `window_size` observations) has drifted from the reference data and 0 otherwise.

* `time`: The number of observations that have been so far passed to the detector as test instances.

* `ert`: The expected run-time the detector was configured to run at in the absence of drift.

* `test_stat`: MMD^2 metric between the reference data and the test_window if `return_test_stat` equals *True*.

* `threshold`: The value the test-statsitic is required to exceed for drift to be detected if `return_test_stat` equals *True*.


```python
preds = cd.predict(x_t, return_test_stat=True)
```

### Managing State

The detector's state may be saved with the `save_state` method:

```python
cd = MMDDriftOnline(x_ref, ert, window_size)  # Instantiate detector at t=0
cd.predict(x_1)  # t=1
cd.save_state('checkpoint_t1')  # Save state at t=1
cd.predict(x_2)  # t=2
```

The previously saved state may then be loaded via the `load_state` method:

```python
# Load state at t=1
cd.load_state('checkpoint_t1')
```

At any point, the state may be reset to `t=0` with the `reset_state` method. When saving the detector with `save_detector`, the state will be saved, unless `t=0` (see [here](../../overview/saving.md#online-detectors)).

## Examples

[Online Drift Detection on the Wine Quality Dataset](../../examples/cd_online_wine.ipynb)

[Online Drift Detection on the Camelyon medical imaging dataset](../../examples/cd_online_camelyon.ipynb)

