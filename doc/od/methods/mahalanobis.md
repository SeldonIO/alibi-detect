---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

[source](../../api/alibi_detect.od.mahalanobis.rst)

# Mahalanobis Distance

## Overview

The Mahalanobis online outlier detector aims to predict anomalies in tabular data. The algorithm calculates an outlier score, which is a measure of distance from the center of the features distribution ([Mahalanobis distance](https://en.wikipedia.org/wiki/Mahalanobis_distance)). If this outlier score is higher than a user-defined threshold, the observation is flagged as an outlier. The algorithm is online, which means that it starts without knowledge about the distribution of the features and learns as requests arrive. Consequently you should expect the output to be bad at the start and to improve over time. The algorithm is suitable for low to medium dimensional tabular data.

The algorithm is also able to include categorical variables. The `fit` step first computes pairwise distances between the categories of each categorical variable. The pairwise distances are based on either the model predictions (*MVDM method*) or the context provided by the other variables in the dataset (*ABDM method*). For MVDM, we use the difference between the conditional model prediction probabilities of each category. This method is based on the Modified Value Difference Metric (MVDM) by [Cost et al (1993)](https://link.springer.com/article/10.1023/A:1022664626993). ABDM stands for Association-Based Distance Metric, a categorical distance measure introduced by [Le et al (2005)](http://www.jaist.ac.jp/~bao/papers/N26.pdf). ABDM infers context from the presence of other variables in the data and computes a dissimilarity measure based on the [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence). Both methods can also be combined as ABDM-MVDM. We can then apply multidimensional scaling to project the pairwise distances into Euclidean space.

## Usage

### Initialize

Parameters:

* `threshold`: Mahalanobis distance threshold above which the instance is flagged as an outlier.

* `n_components`: number of principal components used.

* `std_clip`: feature-wise standard deviation used to clip the observations before updating the mean and covariance matrix.

* `start_clip`: number of observations before clipping is applied.

* `max_n`: algorithm behaves as if it has seen at most `max_n` points.

* `cat_vars`: dictionary with as keys the categorical columns and as values the number of categories per categorical variable. Only needed if categorical variables are present.

* `ohe`: boolean whether the categorical variables are one-hot encoded (OHE) or not. If not OHE, they are assumed to have ordinal encodings.

* `data_type`: can specify data type added to metadata. E.g. *'tabular'* or *'image'*.

Initialized outlier detector example:

```python
from alibi_detect.od import Mahalanobis

od = Mahalanobis(
    threshold=10.,
    n_components=2,
    std_clip=3,
    start_clip=100
)
```

### Fit

We only need to fit the outlier detector if there are categorical variables present in the data. The following parameters can be specified:

* `X`: training batch as a numpy array.

* `y`: model class predictions or ground truth labels for `X`. Used for *'mvdm'* and *'abdm-mvdm'* pairwise distance metrics. Not needed for *'abdm'*.

* `d_type`: pairwise distance metric used for categorical variables. Currently, *'abdm'*, *'mvdm'* and *'abdm-mvdm'* are supported. *'abdm'* infers context from the other variables while *'mvdm'* uses the model predictions. *'abdm-mvdm'* is a weighted combination of the two metrics.

* `w`: weight on *'abdm'* (between 0. and 1.) distance if `d_type` equals *'abdm-mvdm'*.

* `disc_perc`: list with percentiles used in binning of numerical features used for the *'abdm'* and *'abdm-mvdm'* pairwise distance measures.

* `standardize_cat_vars`: standardize numerical values of categorical variables if True.

* `feature_range`: tuple with min and max ranges to allow for numerical values of categorical variables. Min and max ranges can be floats or numpy arrays with dimension *(1, number of features)* for feature-wise ranges.

* `smooth`: smoothing exponent between 0 and 1 for the distances. Lower values will smooth the difference in distance metric between different features.

* `center`: whether to center the scaled distance measures. If False, the min distance for each feature except for the feature with the highest raw max distance will be the lower bound of the feature range, but the upper bound will be below the max feature range.

```python
od.fit(
    X_train,
    d_type='abdm',
    disc_perc=[25, 50, 75]
)
```

It is often hard to find a good threshold value. If we have a batch of normal and outlier data and we know approximately the percentage of normal data in the batch, we can infer a suitable threshold:

```python
od.infer_threshold(
    X, 
    threshold_perc=95
)
```

Beware though that the outlier detector is stateful and every call to the `score` function will update the mean and covariance matrix, even when inferring the threshold.

### Detect

We detect outliers by simply calling `predict` on a batch of instances `X` to compute the instance level Mahalanobis distances. We can also return the instance level outlier score by setting `return_instance_score` to True.

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

[Outlier detection on KDD Cup 99](../../examples/od_mahalanobis_kddcup.ipynb)

