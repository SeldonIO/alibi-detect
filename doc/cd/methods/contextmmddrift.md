---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

[source](../../api/alibi_detect.cd.context_aware.rst)

# Context-Aware Maximum Mean Discrepancy

## Overview

The context-aware maximum mean discrepancy drift detector ([Cobb and Van Looveren, 2022](https://arxiv.org/abs/2203.08644)) is a kernel based method for detecting drift in a manner that can take relevant context into account.  A normal drift detector detects when the distributions underlying two sets of samples $\{x^0_i\}_{i=1}^{n_0}$ and $\{x^1_i\}_{i=1}^{n_1}$ differ. A context-aware drift detector only detects differences that can **not** be attributed to a corresponding difference between sets of associated context variables $\{c^0_i\}_{i=1}^{n_0}$ and $\{c^1_i\}_{i=1}^{n_1}$. 

Context-aware drift detectors afford practitioners the flexibility to specify their desired context variable. It could be a transformation of the data, such as a subset of features, or an unrelated indexing quantity, such as the time or weather. Everything that the practitioner **wishes to allow to change** between the reference window and test window should be captured within the context variable.

On a technical level, the method operates in a manner similar to the [maximum mean discrepancy detector](./mmddrift.ipynb). However, instead of using an estimate of the squared difference between kernel mean embeddings of $X_{\text{ref}}$ and $X_{\text{test}}$ as the test statistic, we now use an estimate of the *expected* squared difference between the kernel [*conditional* mean embeddings](https://arxiv.org/abs/2002.03689) of $X_{\text{ref}}|C$ and $X_{\text{test}}|C$. As well as the kernel defined on the space of data $X$ required to define the test statistic, estimating the statistic additionally requires a kernel defined on the space of the context variable $C$. For any given realisation of the test statistic an associated p-value is then computed using a [conditional permutation test](https://www.jstor.org/stable/2288402).

The detector is designed for cases where the training data contains a rich variety of contexts and individual test windows may cover a much more limited subset. **It is assumed that the test contexts remain within the support of those observed in the reference set**.

## Usage

### Initialize


Arguments:

* `x_ref`: Data used as reference distribution.
* `c_ref`: Context for the reference distribution.


Keyword arguments:

* `backend`: Both **TensorFlow** and **PyTorch** implementations of the context-aware MMD detector as well as various preprocessing steps are available. Specify the backend (*tensorflow* or *pytorch*). Defaults to *tensorflow*.

* `p_val`: p-value used for significance of the permutation test.

* `preprocess_at_init`: Whether to already apply the (optional) preprocessing step to the reference data `x_ref` at initialization and store the preprocessed data. Dependent on the preprocessing step, this can reduce the computation time for the predict step significantly, especially when the reference dataset is large. Defaults to *True*. It is possible that it needs to be set to *False* if the preprocessing step requires statistics from both the reference and test data, such as the mean or standard deviation.

* `update_ref`: Reference data can optionally be updated to the last N instances seen by the detector. The parameter should be passed as a dictionary *{'last': N}*.

* `preprocess_fn`: Function to preprocess the data (`x_ref` and `x`) before computing the data drift metrics. Typically a dimensionality reduction technique. **NOTE**: Preprocessing is not applied to the context data.

* `x_kernel`: Kernel defined on the data `x_*`. Defaults to a Gaussian RBF kernel (`from alibi_detect.utils.pytorch import GaussianRBF` or `from alibi_detect.utils.tensorflow import GaussianRBF` dependent on the backend used).

* `c_kernel`: Kernel defined on the context `c_*`. Defaults to a Gaussian RBF kernel (`from alibi_detect.utils.pytorch import GaussianRBF` or `from alibi_detect.utils.tensorflow import GaussianRBF` dependent on the backend used).

* `n_permutations`: Number of permutations used in the conditional permutation test.

* `prop_c_held`: Proportion of contexts held out to condition on.

* `n_folds`: Number of cross-validation folds used when tuning the regularisation parameters.

* `batch_size`:  If not `None`, then compute batches of MMDs at a time rather than all at once which could lead to memory issues.

* `input_shape`: Optionally pass the shape of the input data.

* `data_type`: can specify data type added to the metadata. E.g. *'tabular'* or *'image'*.

* `verbose`: Whether or not to print progress during configuration.


Additional PyTorch keyword arguments:

* `device`: *cuda* or *gpu* to use the GPU and *cpu* for the CPU. If the device is not specified, the detector will try to leverage the GPU if possible and otherwise fall back on CPU.


Initialized drift detector example with the PyTorch backend:


```python
from alibi_detect.cd import ContextMMDDrift

cd = ContextMMDDrift(x_ref, c_ref, p_val=.05, backend='pytorch')
```

The same detector in TensorFlow:

```python
from alibi_detect.cd import ContextMMDDrift

cd = ContextMMDDrift(x_ref, c_ref, p_val=.05, backend='tensorflow')
```

### Detect Drift

We detect data drift by simply calling `predict` on a batch of test or deployment instances `x` and contexts `c`. We can return the p-value and the threshold of the permutation test by setting `return_p_val` to *True* and the context-aware maximum mean discrepancy metric and threshold by setting `return_distance` to *True*. We can also set `return_coupling` to *True* which additionally returns the coupling matrices $W_\text{ref,test}$, $W_\text{ref,ref}$ and $W_\text{test,test}$. As illustrated in the examples ([text](../../examples/cd_context_20newsgroup.ipynb), [ECGs](../../examples/cd_context_ecg.ipynb)) this can provide deep insights into where the reference and test distributions are similar and where they differ.

The prediction takes the form of a dictionary with `meta` and `data` keys. `meta` contains the detector's metadata while `data` is also a dictionary which contains the actual predictions stored in the following keys:

* `is_drift`: 1 if the sample tested has drifted from the reference data and 0 otherwise.

* `p_val`: contains the p-value if `return_p_val` equals *True*.

* `threshold`: p-value threshold if `return_p_val` equals *True*.

* `distance`: conditional MMD^2 metric between the reference data and the new batch if `return_distance` equals *True*.

* `distance_threshold`: conditional MMD^2 metric value from the permutation test which corresponds to the the p-value threshold.

* `coupling_xx`: coupling matrix $W_\text{ref,ref}$ for the reference data.

* `coupling_yy`: coupling matrix $W_\text{test,test}$ for the test data.

* `coupling_xy`: coupling matrix $W_\text{ref,test}$ between the reference and test data.


```python
preds = cd.predict(x, c, return_p_val=True, return_distance=True, return_coupling=True)
```

## Examples


### Text

[Context-aware drift detection on news articles](../../examples/cd_context_20newsgroup.ipynb)

### Time series

[Context-aware drift detection on ECGs](../../examples/cd_context_ecg.ipynb)

