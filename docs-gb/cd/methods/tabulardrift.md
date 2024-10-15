---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

[source](../../api/alibi_detect.cd.tabular.rst)

# Mixed-type tabular data

## Overview

The drift detector applies feature-wise two-sample [Kolmogorov-Smirnov](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test) (K-S) tests for the continuous numerical features and [Chi-Squared](https://en.wikipedia.org/wiki/Chi-squared_test) tests for the categorical features. For multivariate data, the obtained p-values for each feature are aggregated either via the [Bonferroni](https://mathworld.wolfram.com/BonferroniCorrection.html) or the [False Discovery Rate](http://www.math.tau.ac.il/~ybenja/MyPapers/benjamini_hochberg1995.pdf) (FDR) correction. The Bonferroni correction is more conservative and controls for the probability of at least one false positive. The FDR correction on the other hand allows for an expected fraction of false positives to occur. Similarly to the other drift detectors, a preprocessing steps could be applied, but the output features need to be categorical.

## Usage

### Initialize

Arguments:

* `x_ref`: Data used as reference distribution.


Keyword arguments:

* `p_val`: p-value used for significance of the K-S and Chi-Squared test across all features. If the FDR correction method is used, this corresponds to the acceptable q-value.

* `categories_per_feature`: Dictionary with as keys the column indices of the categorical features and optionally as values the number of possible categorical values for that feature or a list with the possible values. If you know which features are categorical and simply want to infer the possible values of the categorical feature from the reference data you can pass a *Dict[int, NoneType]* such as *{0: None, 3: None}* if features 0 and 3 are categorical. If you also know how many categories are present for a given feature you could pass this in the `categories_per_feature` dict in the *Dict[int, int]* format, e.g. *{0: 3, 3: 2}*. If you pass N categories this will assume the possible values for the feature are [0, ..., N-1]. You can also explicitly pass the possible categories in the *Dict[int, List[int]]* format, e.g. *{0: [0, 1, 2], 3: [0, 55]}*. Note that the categories can be arbitrary *int* values.

* `preprocess_at_init`: Whether to already apply the (optional) preprocessing step to the reference data at initialization and store the preprocessed data. Dependent on the preprocessing step, this can reduce the computation time for the predict step significantly, especially when the reference dataset is large. Defaults to *True*. It is possible that it needs to be set to *False* if the preprocessing step requires statistics from both the reference and test data, such as the mean or standard deviation.

* `x_ref_preprocessed`: Whether or not the reference data `x_ref` has already been preprocessed. If *True*, the reference data will be skipped and preprocessing will only be applied to the test data passed to `predict`.

* `update_x_ref`: Reference data can optionally be updated to the last N instances seen by the detector or via [reservoir sampling](https://en.wikipedia.org/wiki/Reservoir_sampling) with size N. For the former, the parameter equals *{'last': N}* while for reservoir sampling *{'reservoir_sampling': N}* is passed.

* `preprocess_fn`: Function to preprocess the data before computing the data drift metrics. Typically a dimensionality reduction technique.

* `correction`: Correction type for multivariate data. Either *'bonferroni'* or *'fdr'* (False Discovery Rate).

* `alternative`: Defines the alternative hypothesis for the K-S tests. Options are *'two-sided'* (default), *'less'* or *'greater'*. Make sure to use *'two-sided'* when mixing categorical and numerical features.

* `n_features`: Number of features used in the K-S and Chi-Squared tests. No need to pass it if no preprocessing takes place. In case of a preprocessing step, this can also be inferred automatically but could be more expensive to compute.

* `data_type`: can specify data type added to metadata. E.g. *'tabular'*.

Initialized drift detector example:

```python
from alibi_detect.cd import TabularDrift

cd = TabularDrift(x_ref, p_val=0.05, categories_per_feature={0: None, 3: None})
```

### Detect Drift

We detect data drift by simply calling `predict` on a batch of instances `x`. We can return the feature-wise p-values before the multivariate correction by setting `return_p_val` to *True*. The drift can also be detected at the feature level by setting `drift_type` to *'feature'*. No multivariate correction will take place since we return the output of *n_features* univariate tests. For drift detection on all the features combined with the correction, use *'batch'*. `return_p_val` equal to *True* will also return the threshold used by the detector (either for the univariate case or after the multivariate correction).

The prediction takes the form of a dictionary with `meta` and `data` keys. `meta` contains the detector's metadata while `data` is also a dictionary which contains the actual predictions stored in the following keys:

* `is_drift`: 1 if the sample tested has drifted from the reference data and 0 otherwise.

* `p_val`: contains feature-level p-values if `return_p_val` equals *True*.

* `threshold`: for feature-level drift detection the threshold equals the p-value used for the significance of the K-S and Chi-Squared tests. Otherwise the threshold after the multivariate correction (either *bonferroni* or *fdr*) is returned.

* `distance`: feature-wise K-S or Chi-Squared statistics between the reference data and the new batch if `return_distance` equals *True*.


```python
preds = cd.predict(x, drift_type='batch', return_p_val=True, return_distance=True)
```

## Examples

[Drift detection on income prediction](../../examples/cd_chi2ks_adult.ipynb)

