---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

[source](../../api/alibi_detect.od.isolationforest.rst)

# Isolation Forest

## Overview

[Isolation forests](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf) (IF) are tree based models specifically used for outlier detection. The IF isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature. The number of splittings required to isolate a sample is equivalent to the path length from the root node to the terminating node. This path length, averaged over a forest of random trees, is a measure of normality and is used to define an anomaly score. Outliers can typically be isolated quicker, leading to shorter paths. The algorithm is suitable for low to medium dimensional tabular data.

## Usage

### Initialize

Parameters:

* `threshold`: threshold value for the outlier score above which the instance is flagged as an outlier.

* `n_estimators`: number of base estimators in the ensemble. Defaults to 100.

* `max_samples`: number of samples to draw from the training data to train each base estimator. If *int*, draw `max_samples` samples. If *float*, draw `max_samples` *times number of features* samples. If *'auto'*, `max_samples` = min(256, number of samples).

* `max_features`: number of features to draw from the training data to train each base estimator. If *int*, draw `max_features` features. If float, draw `max_features` *times number of features* features.

* `bootstrap`: whether to fit individual trees on random subsets of the training data, sampled with replacement.

* `n_jobs`: number of jobs to run in parallel for `fit` and `predict`.

* `data_type`: can specify data type added to metadata. E.g. *'tabular'* or *'image'*.

Initialized outlier detector example:

```python
from alibi_detect.od import IForest

od = IForest(
    threshold=0.,
    n_estimators=100
)
```

### Fit

We then need to train the outlier detector. The following parameters can be specified:

* `X`: training batch as a numpy array.

* `sample_weight`: array with shape *(batch size,)* used to assign different weights to each instance during training. Defaults to *None*.

```python
od.fit(
    X_train
)
```

It is often hard to find a good threshold value. If we have a batch of normal and outlier data and we know approximately the percentage of normal data in the batch, we can infer a suitable threshold:

```python
od.infer_threshold(
    X, 
    threshold_perc=95
)
```

### Detect

We detect outliers by simply calling `predict` on a batch of instances `X` to compute the instance level outlier scores. We can also return the instance level outlier score by setting `return_instance_score` to True.

The prediction takes the form of a dictionary with `meta` and `data` keys. `meta` contains the detector's metadata while `data` is also a dictionary which contains the actual predictions stored in the following keys:

* `is_outlier`: boolean whether instances are above the threshold and therefore outlier instances. The array is of shape *(batch size,)*.

* `instance_score`: contains instance level scores if `return_instance_score` equals True.


```python
preds = od.predict(
    X,
    return_instance_score=True
)
```

## Examples

### Tabular

[Outlier detection on KDD Cup 99](../../examples/od_if_kddcup.ipynb)

