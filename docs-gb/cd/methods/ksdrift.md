---
jupyter:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

[source](../../api/alibi_detect.cd.ks.rst)

# Kolmogorov-Smirnov

## Overview

The drift detector applies feature-wise two-sample [Kolmogorov-Smirnov](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test) (K-S) tests. For multivariate data, the obtained p-values for each feature are aggregated either via the [Bonferroni](https://mathworld.wolfram.com/BonferroniCorrection.html) or the [False Discovery Rate](http://www.math.tau.ac.il/~ybenja/MyPapers/benjamini_hochberg1995.pdf) (FDR) correction. The Bonferroni correction is more conservative and controls for the probability of at least one false positive. The FDR correction on the other hand allows for an expected fraction of false positives to occur.

For high-dimensional data, we typically want to reduce the dimensionality before computing the feature-wise univariate K-S tests and aggregating those via the chosen correction method. Following suggestions in [Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift](https://arxiv.org/abs/1810.11953), we incorporate Untrained AutoEncoders (UAE) and black-box shift detection using the classifier's softmax outputs ([BBSDs](https://arxiv.org/abs/1802.03916)) as out-of-the box preprocessing methods and note that [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) can also be easily implemented using `scikit-learn`. Preprocessing methods which do not rely on the classifier will usually pick up drift in the input data, while BBSDs focuses on label shift. The [adversarial detector](https://arxiv.org/abs/2002.09364) which is part of the library can also be transformed into a drift detector picking up drift that reduces the performance of the classification model. We can therefore combine different preprocessing techniques to figure out if there is drift which hurts the model performance, and whether this drift can be classified as input drift or label shift.

Detecting input data drift (covariate shift) $\Delta p(x)$ for text data requires a custom preprocessing step. We can pick up changes in the semantics of the input by extracting (contextual) embeddings and detect drift on those. Strictly speaking we are not detecting $\Delta p(x)$ anymore since the whole training procedure (objective function, training data etc) for the (pre)trained embeddings has an impact on the embeddings we extract. The library contains functionality to leverage pre-trained embeddings from [HuggingFace's transformer package](https://github.com/huggingface/transformers) but also allows you to easily use your own embeddings of choice. Both options are illustrated with examples in the [Text drift detection on IMDB movie reviews](../../examples/cd_text_imdb.ipynb) notebook.

## Usage

### Initialize

Arguments:

* `x_ref`: Data used as reference distribution.

Keyword arguments:

* `p_val`: p-value used for significance of the K-S test. If the FDR correction method is used, this corresponds to the acceptable q-value.

* `preprocess_at_init`: Whether to already apply the (optional) preprocessing step to the reference data at initialization and store the preprocessed data. Dependent on the preprocessing step, this can reduce the computation time for the predict step significantly, especially when the reference dataset is large. Defaults to *True*. It is possible that it needs to be set to *False* if the preprocessing step requires statistics from both the reference and test data, such as the mean or standard deviation.

* `x_ref_preprocessed`: Whether or not the reference data `x_ref` has already been preprocessed. If *True*, the reference data will be skipped and preprocessing will only be applied to the test data passed to `predict`.

* `update_x_ref`: Reference data can optionally be updated to the last N instances seen by the detector or via [reservoir sampling](https://en.wikipedia.org/wiki/Reservoir_sampling) with size N. For the former, the parameter equals *{'last': N}* while for reservoir sampling *{'reservoir_sampling': N}* is passed.

* `preprocess_fn`: Function to preprocess the data before computing the data drift metrics. Typically a dimensionality reduction technique.

* `correction`: Correction type for multivariate data. Either *'bonferroni'* or *'fdr'* (False Discovery Rate).

* `alternative`: Defines the alternative hypothesis. Options are *'two-sided'* (default), *'less'* or *'greater'*.

* `n_features`: Number of features used in the K-S test. No need to pass it if no preprocessing takes place. In case of a preprocessing step, this can also be inferred automatically but could be more expensive to compute.

* `input_shape`: Shape of input data.

* `data_type`: can specify data type added to metadata. E.g. *'tabular'* or *'image'*.

Initialized drift detector example:

```python
from alibi_detect.cd import KSDrift

cd = KSDrift(x_ref, p_val=0.05)
```

### Detect Drift

We detect data drift by simply calling `predict` on a batch of instances `x`. We can return the feature-wise p-values before the multivariate correction by setting `return_p_val` to *True*. The drift can also be detected at the feature level by setting `drift_type` to *'feature'*. No multivariate correction will take place since we return the output of *n_features* univariate tests. For drift detection on all the features combined with the correction, use *'batch'*. `return_p_val` equal to *True* will also return the threshold used by the detector (either for the univariate case or after the multivariate correction).

The prediction takes the form of a dictionary with `meta` and `data` keys. `meta` contains the detector's metadata while `data` is also a dictionary which contains the actual predictions stored in the following keys:

* `is_drift`: 1 if the sample tested has drifted from the reference data and 0 otherwise.

* `p_val`: contains feature-level p-values if `return_p_val` equals *True*.

* `threshold`: for feature-level drift detection the threshold equals the p-value used for the significance of the K-S test. Otherwise the threshold after the multivariate correction (either *bonferroni* or *fdr*) is returned.

* `distance`: feature-wise K-S statistics between the reference data and the new batch if `return_distance` equals *True*.


```python
preds = cd.predict(x, drift_type='batch', return_p_val=True, return_distance=True)
```

## Examples

### Graph

[Drift detection on molecular graphs](../../examples/cd_mol.ipynb)

### Image

[Drift detection on CIFAR10](../../examples/cd_ks_cifar10.ipynb)


### Text

[Text drift detection on IMDB movie reviews](../../examples/cd_text_imdb.ipynb)

