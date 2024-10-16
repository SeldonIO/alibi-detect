---
jupyter:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

[source](../../api/alibi_detect.cd.lsdd.rst)

# Least-Squares Density Difference

## Overview

The [least-squares density difference detector](https://alippi.faculty.polimi.it/articoli/A%20Pdf%20free%20Change%20Detection%20Test%20Based%20on%20Density%20Difference%20Estimation.pdf) is a method for multivariate 2 sample testing. The LSDD between two distributions $p$ and $q$ on $\mathcal{X}$ is defined as $$LSDD(p,q) = \int_{\mathcal{X}} (p(x)-q(x))^2 \,dx.$$

Given two samples we can compute an estimate of the $LSDD$ between the two underlying distributions and use it as a test statistic. We then obtain a $p$-value via a [permutation test](https://en.wikipedia.org/wiki/Resampling_(statistics)) on the values of the $LSDD$ estimates. In practice we actually estimate the LSDD scaled by a factor that maintains numerical stability when dimensionality is high.

<div class="alert alert-info">
Note

$LSDD$ is based on the assumption that a probability density exists for both distributions and hence is only suitable for continuous data. If you are working with tabular data containing categorical variables, we recommend using the [TabularDrift detector](../methods/tabulardrift.ipynb) instead.

</div>


For high-dimensional data, we typically want to reduce the dimensionality before computing the permutation test. Following suggestions in [Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift](https://arxiv.org/abs/1810.11953), we incorporate Untrained AutoEncoders (UAE) and black-box shift detection using the classifier's softmax outputs ([BBSDs](https://arxiv.org/abs/1802.03916)) as out-of-the box preprocessing methods and note that [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) can also be easily implemented using `scikit-learn`. Preprocessing methods which do not rely on the classifier will usually pick up drift in the input data, while BBSDs focuses on label shift.

Detecting input data drift (covariate shift) $\Delta p(x)$ for text data requires a custom preprocessing step. We can pick up changes in the semantics of the input by extracting (contextual) embeddings and detect drift on those. Strictly speaking we are not detecting $\Delta p(x)$ anymore since the whole training procedure (objective function, training data etc) for the (pre)trained embeddings has an impact on the embeddings we extract. The library contains functionality to leverage pre-trained embeddings from [HuggingFace's transformer package](https://github.com/huggingface/transformers) but also allows you to easily use your own embeddings of choice. Both options are illustrated with examples in the [Text drift detection on IMDB movie reviews](../../examples/cd_text_imdb.ipynb) notebook.

## Usage

### Initialize


Arguments:

* `x_ref`: Data used as reference distribution.


Keyword arguments:

* `backend`: Both **TensorFlow** and **PyTorch** implementations of the LSD detector as well as various preprocessing steps are available. Specify the backend (*tensorflow* or *pytorch*). Defaults to *tensorflow*.

* `p_val`: p-value used for significance of the permutation test.

* `preprocess_at_init`: Whether to already apply the (optional) preprocessing step to the reference data at initialization and store the preprocessed data. Dependent on the preprocessing step, this can reduce the computation time for the predict step significantly, especially when the reference dataset is large. Defaults to *True*. It is possible that it needs to be set to *False* if the preprocessing step requires statistics from both the reference and test data, such as the mean or standard deviation.

* `x_ref_preprocessed`: Whether or not the reference data `x_ref` has already been preprocessed. If *True*, the reference data will be skipped and preprocessing will only be applied to the test data passed to `predict`.

* `update_x_ref`: Reference data can optionally be updated to the last N instances seen by the detector or via [reservoir sampling](https://en.wikipedia.org/wiki/Reservoir_sampling) with size N. For the former, the parameter equals *{'last': N}* while for reservoir sampling *{'reservoir_sampling': N}* is passed.

* `preprocess_fn`: Function to preprocess the data before computing the data drift metrics. Typically a dimensionality reduction technique.

* `sigma`: Optionally set the bandwidth of the Gaussian kernel used in estimating the LSDD. Can also pass multiple bandwidth values as an array. The kernel evaluation is then averaged over those bandwidths.  If `sigma` is not specified, the 'median heuristic' is adopted whereby `sigma` is set as the median pairwise distance between reference samples.

* `n_permutations`: Number of permutations used in the permutation test.

* `n_kernel_centers`: The number of reference samples to use as centers in the Gaussian kernel model used to estimate LSDD. Defaults to 1/20th of the reference data.

* `lambda_rd_max`: The maximum relative difference between two estimates of LSDD that the regularization parameter lambda is allowed to cause. Defaults to 0.2 as in the paper.

* `input_shape`: Optionally pass the shape of the input data.

* `data_type`: can specify data type added to the metadata. E.g. *'tabular'* or *'image'*.


Additional PyTorch keyword arguments:

* `device`: *cuda* or *gpu* to use the GPU and *cpu* for the CPU. If the device is not specified, the detector will try to leverage the GPU if possible and otherwise fall back on CPU.


Initialized drift detector example:


```python
from alibi_detect.cd import LSDDDrift

cd = LSDDDrift(x_ref, backend='tensorflow', p_val=.05)
```

The same detector in PyTorch:

```python
cd = LSDDDrift(x_ref, backend='pytorch', p_val=.05)
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

cd = LSDDDrift(x_ref, backend='pytorch', p_val=.05, preprocess_fn=preprocess_fn)
```
The same functionality is supported in TensorFlow and the main difference is that you would import from `alibi_detect.cd.tensorflow import preprocess_drift`. Other preprocessing steps such as the output of hidden layers of a model or extracted text embeddings using transformer models can be used in a similar way in both frameworks. TensorFlow example for the hidden layer output:

```python
from alibi_detect.cd.tensorflow import HiddenOutput, preprocess_drift

model = # TensorFlow model; tf.keras.Model or tf.keras.Sequential
preprocess_fn = partial(preprocess_drift, model=HiddenOutput(model, layer=-1), batch_size=128)

cd = LSDDDrift(x_ref, backend='tensorflow', p_val=.05, preprocess_fn=preprocess_fn)
```

The `LSDDDrift` detector can be used in exactly the same way as the `MMDDrift` detector which is further demonstrated in the [Drift detection on CIFAR10](../../examples/cd_mmd_cifar10.ipynb) example.

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
cd = LSDDDrift(x_ref, backend='pytorch', p_val=.05, preprocess_fn=preprocess_fn)
```

Again the same functionality is supported in TensorFlow but with `from alibi_detect.cd.tensorflow import preprocess_drift` and `from alibi_detect.models.tensorflow import TransformerEmbedding` imports. Check out the [Text drift detection on IMDB movie reviews](../../examples/cd_text_imdb.ipynb) example for more information.

### Detect Drift

We detect data drift by simply calling `predict` on a batch of instances `x`. We can return the p-value and the threshold of the permutation test by setting `return_p_val` to *True* and the maximum mean discrepancy metric and threshold by setting `return_distance` to *True*.

The prediction takes the form of a dictionary with `meta` and `data` keys. `meta` contains the detector's metadata while `data` is also a dictionary which contains the actual predictions stored in the following keys:

* `is_drift`: 1 if the sample tested has drifted from the reference data and 0 otherwise.

* `p_val`: contains the p-value if `return_p_val` equals *True*.

* `threshold`: p-value threshold if `return_p_val` equals *True*.

* `distance`: LSDD metric between the reference data and the new batch if `return_distance` equals *True*.

* `distance_threshold`: LSDD metric value from the permutation test which corresponds to the the p-value threshold.


```python
preds = cd.predict(X, return_p_val=True, return_distance=True)
```

## Examples

For the related `MMDDrift` detector.

### Image

[Drift detection on CIFAR10](../../examples/cd_mmd_cifar10.ipynb)

### Text

[Text drift detection on IMDB movie reviews](../../examples/cd_text_imdb.ipynb)

