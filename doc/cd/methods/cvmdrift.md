---
jupyter:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

[source](../../api/alibi_detect.cd.cvm.rst)

# Cramér-von Mises

## Overview

The CVM drift detector is a non-parametric drift detector, which applies feature-wise two-sample [Cramér-von Mises](https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93von_Mises_criterion) (CVM) tests. For two empirical distributions $F(z)$ and $F_{ref}(z)$, the CVM test statistic is defined as

$$
W = \sum_{z\in k} \left| F(z) - F_{ref}(z) \right|^2,
$$

where $k$ is the joint sample. The CVM test is an alternative to the [Kolmogorov-Smirnov](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test) (K-S) two-sample test, which uses the maximum distance between two emphirical distributions $F(z)$ and $F_{ref}(z)$. By using the full joint sample, the CVM can exhibit greater power against shifts in higher moments, such as variance changes.


For multivariate data, the detector applies a separate CVM test to each feature, and the p-values obtained for each feature are aggregated either via the [Bonferroni](https://mathworld.wolfram.com/BonferroniCorrection.html) or the [False Discovery Rate](http://www.math.tau.ac.il/~ybenja/MyPapers/benjamini_hochberg1995.pdf) (FDR) correction. The Bonferroni correction is more conservative and controls for the probability of at least one false positive. The FDR correction on the other hand allows for an expected fraction of false positives to occur. As with other univariate detectors such as the [Kolmogorov-Smirnov](ksdrift.ipynb) detector, for high-dimensional data, we typically want to reduce the dimensionality before computing the feature-wise univariate FET tests and aggregating those via the chosen correction method. See [Dimension Reduction](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/background.html#dimension-reduction) for more guidance on this.

## Usage

### Initialize

Arguments:

* `x_ref`: Data used as reference distribution.

Keyword arguments:

* `p_val`: p-value used for significance of the CVM test. If the FDR correction method is used, this corresponds to the acceptable q-value.

* `preprocess_at_init`: Whether to already apply the (optional) preprocessing step to the reference data at initialization and store the preprocessed data. Dependent on the preprocessing step, this can reduce the computation time for the predict step significantly, especially when the reference dataset is large. Defaults to *True*. It is possible that it needs to be set to *False* if the preprocessing step requires statistics from both the reference and test data, such as the mean or standard deviation.

* `x_ref_preprocessed`: Whether or not the reference data `x_ref` has already been preprocessed. If *True*, the reference data will be skipped and preprocessing will only be applied to the test data passed to `predict`.

* `update_x_ref`: Reference data can optionally be updated to the last N instances seen by the detector or via [reservoir sampling](https://en.wikipedia.org/wiki/Reservoir_sampling) with size N. For the former, the parameter equals *{'last': N}* while for reservoir sampling *{'reservoir_sampling': N}* is passed.

* `preprocess_fn`: Function to preprocess the data before computing the data drift metrics.

* `correction`: Correction type for multivariate data. Either *'bonferroni'* or *'fdr'* (False Discovery Rate).

* `n_features`: Number of features used in the CVM test. No need to pass it if no preprocessing takes place. In case of a preprocessing step, this can also be inferred automatically but could be more expensive to compute.

* `input_shape`: Shape of input data.

* `data_type`: can specify data type added to metadata. E.g. *'tabular'* or *'image'*.

Initialized drift detector example:

```python
from alibi_detect.cd import CVMDrift

cd = CVMDrift(x_ref, p_val=0.05)
```

### Detect Drift

We detect data drift by simply calling `predict` on a batch of instances `x`. We can return the feature-wise p-values before the multivariate correction by setting `return_p_val` to *True*. The drift can also be detected at the feature level by setting `drift_type` to *'feature'*. No multivariate correction will take place since we return the output of *n_features* univariate tests. For drift detection on all the features combined with the correction, use *'batch'*. `return_p_val` equal to *True* will also return the threshold used by the detector (either for the univariate case or after the multivariate correction).

The prediction takes the form of a dictionary with `meta` and `data` keys. `meta` contains the detector's metadata while `data` is also a dictionary which contains the actual predictions stored in the following keys:

* `is_drift`: 1 if the sample tested has drifted from the reference data and 0 otherwise.

* `p_val`: contains feature-level p-values if `return_p_val` equals *True*.

* `threshold`: for feature-level drift detection the threshold equals the p-value used for the significance of the CVM test. Otherwise the threshold after the multivariate correction (either *bonferroni* or *fdr*) is returned.

* `distance`: feature-wise CVM statistics between the reference data and the new batch if `return_distance` equals *True*.


```python
preds = cd.predict(x, drift_type='batch', return_p_val=True, return_distance=True)
```

## Examples

[Supervised drift detection on the penguins dataset](../../examples/cd_supervised_penguins.ipynb)

