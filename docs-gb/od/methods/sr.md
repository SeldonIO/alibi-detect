---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

[source](../../api/alibi_detect.od.sr.rst)

# Spectral Residual

## Overview

The Spectral Residual outlier detector is based on the paper [Time-Series Anomaly Detection Service at Microsoft](https://arxiv.org/abs/1906.03821) and is suitable for **unsupervised online anomaly detection in univariate time series** data. The algorithm first computes the [Fourier Transform](https://en.wikipedia.org/wiki/Fourier_transform) of the original data. Then it computes the *spectral residual* of the log amplitude of the transformed signal before applying the Inverse Fourier Transform to map the sequence back from the frequency to the time domain. This sequence is called the *saliency map*. The anomaly score is then computed as the relative difference between the saliency map values and their moving averages. If the score is above a threshold, the value at a specific timestep is flagged as an outlier. For more details, please check out the [paper](https://arxiv.org/abs/1906.03821).

## Usage

### Initialize

Parameters:

* `threshold`: Threshold used to classify outliers. Relative saliency map distance from the moving average.

* `window_amp`: Window used for the moving average in the *spectral residual* computation. The spectral residual is the difference between the log amplitude of the Fourier Transform and a convolution of the log amplitude over `window_amp`.

* `window_local`: Window used for the moving average in the outlier score computation. The outlier score computes the relative difference between the saliency map and a moving average of the saliency map over `window_local` timesteps.

* `padding_amp_method`:
    Padding method to be used prior to each convolution over log amplitude.
    Possible values: `constant` | `replicate` | `reflect`. Default value: `replicate`.

     - `constant` - padding with constant 0.

     - `replicate` - repeats the last/extreme value.

     - `reflect` - reflects the time series.

* `padding_local_method`:
    Padding method to be used prior to each convolution over saliency map.
    Possible values: `constant` | `replicate` | `reflect`. Default value: `replicate`.

     - `constant` - padding with constant 0.

     - `replicate` - repeats the last/extreme value.

     - `reflect` - reflects the time series.

* `padding_amp_side`:
    Whether to pad the amplitudes on both sides or only on one side.
    Possible values: `bilateral` | `left` | `right`.

* `n_est_points`: Number of estimated points padded to the end of the sequence.

* `n_grad_points`: Number of points used for the gradient estimation of the additional points padded to the end of the sequence. The paper sets this value to 5.

Initialized outlier detector example:

```python
from alibi_detect.od import SpectralResidual

od = SpectralResidual(
    threshold=1.,
    window_amp=20,
    window_local=20,
    padding_amp_method='reflect',
    padding_local_method='reflect',
    padding_amp_side='bilateral',
    n_est_points=10,
    n_grad_points=5
)
```

It is often hard to find a good threshold value. If we have a time series containing both normal and outlier data and we know approximately the percentage of normal data in the time series, we can infer a suitable threshold:

```python
od.infer_threshold(
    X,
    t=t,  # array with timesteps, assumes dt=1 between observations if omitted
    threshold_perc=95
)
```

### Detect

We detect outliers by simply calling `predict` on a time series `X` to compute the outlier scores and flag the anomalies. We can also return the instance (timestep) level outlier score by setting `return_instance_score` to True.

The prediction takes the form of a dictionary with `meta` and `data` keys. `meta` contains the detector's metadata while `data` is also a dictionary which contains the actual predictions stored in the following keys:

* `is_outlier`: boolean whether instances are above the threshold and therefore outlier instances. The array is of shape *(timesteps,)*.

* `instance_score`: contains instance level scores if `return_instance_score` equals True.


```python
preds = od.predict(
    X,
    t=t,  # array with timesteps, assumes dt=1 between observations if omitted
    return_instance_score=True
)
```

## Examples

[Time series outlier detection with Spectral Residuals on synthetic data](../../examples/od_sr_synth.ipynb)

