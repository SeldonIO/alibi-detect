---
jupyter:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

[source](../../api/alibi_detect.cd.fet_online.rst)

# Online Fisher's Exact Test

## Overview

The online [Fisher's Exact Test](https://en.wikipedia.org/wiki/Fisher%27s_exact_test) detector is a non-parametric method for online drift detection. Like the [offline Fisher's Exact Test](fetdrift.ipynb) detector, it applies an Fisher's Exact Test (FET) to each feature. It is intended for application to [Bernoulli](https://en.wikipedia.org/wiki/Bernoulli_distribution) streams, with binary data consisting of either `(True, False)` or `(0, 1)`. This detector is ideal for use in a supervised setting, monitoring drift in a model's instance level accuracy (i.e. correct prediction = 0, and incorrect prediction = 1). 

### Threshold configuration
Online detectors assume the reference data is large and fixed and operate on single data points at a time (rather than batches). These data points are passed into the test-windows, and a two-sample test-statistic (in this case $F=1-\hat{p}$) between the reference data and test-window is computed at each time-step. When the test-statistic exceeds a preconfigured threshold, drift is detected. Configuration of the thresholds requires specification of the expected run-time (ERT) which specifies how many time-steps that the detector, on average, should run for in the absence of drift before making a false detection. 

In a similar manner to that proposed in [this paper](https://arxiv.org/pdf/1212.6020.pdf) by Ross et al. <cite data-cite="Ross2012b"><!-- --></cite>, thresholds are configured by simulating `n_bootstraps` Bernoulli streams. The length of streams can be set with the `t_max` parameter. Since the thresholds are expected to converge after `t_max = 2*max(window_sizes) - 1` time steps, we only need to simulate trajectories and estimate thresholds up to this point, and `t_max` is set to this value by default. Following [[1]](#References), the test statistics are smoothed using an exponential moving average to remove their discreteness, allowing more precise quantiles to be targeted:

$$
F_t = (1-\lambda)F_{t-1} + \lambda F_t
$$

For a window size of $W$, at time $t$ the value of the statistic $F_t$ depends on more than just the previous $W$ values. If $\lambda$, set by `lam`, is too small, thresholds may keep decreasing well past $2W - 1$ timesteps. To avoid this, the default `lam` is set to a high value of $\lambda=0.99$, meaning that discreteness is still broken, but the value of the test statistic depends (almost) solely on the last $W$ observations. If more smoothing is desired, the `t_max` parameter can be manually set at a larger value.

    
<div class="alert alert-info">

**Note**
    
The detector must configure thresholds for each window size and each feature. This can be a time consuming process if the number of features is high. For high-dimensional data users are recommended to apply a dimension reduction step via `preprocess_fn`. 
</div>


### Window sizes
Specification of test-window sizes (the detector accepts multiple windows of different size $W$) is also required, with smaller windows allowing faster response to severe drift and larger windows allowing more power to detect slight drift. Since this detector requires a window to be full to function, the ERT is measured from `t = min(window_sizes)-1`.

### Multivariate data
Although this detector is primarly intended for univariate data, it can also be applied to multivariate data. In this case, the detector makes a correction similar to the Bonferroni correction used for the offline detector. Given $d$ features, the detector configures thresholds by targeting the $1-\beta$ quantile of test statistics over the simulated streams, where $\beta = 1 - (1-(1/ERT))^{(1/d)}$. For the univariate case, this simplifies to $\beta = 1/ERT$. At prediction time, drift is flagged if the test statistic of any feature stream exceed the thresholds.


<div class="alert alert-info">

**Note**
    
In the multivariate case, for the ERT to be accurately targeted the feature streams must be independent.

</div>

## Usage

### Initialize


Arguments:

* `x_ref`: Data used as reference distribution.
* `ert`: The expected run-time in the absence of drift, starting from *t=min(windows_sizes)*.
* `window_sizes`: The sizes of the sliding test-windows used to compute the test-statistics. Smaller windows focus on responding quickly to severe drift, larger windows focus on ability to detect slight drift.


Keyword arguments:

* `preprocess_fn`: Function to preprocess the data before computing the data drift metrics.
* `n_bootstraps`: The number of bootstrap simulations used to configure the thresholds. The larger this is the more accurately the desired ERT will be targeted. Should ideally be at least an order of magnitude larger than the ERT.
* `t_max`: Length of streams to simulate when configuring thresholds. If *None*, this is set to 2 * max(`window_sizes`) - 1.
* `alternative`: Defines the alternative hypothesis. Options are *'greater'* (default) or *'less'*, corresponding to an increase or decrease in the mean of the Bernoulli stream. 
* `lam`: Smoothing coefficient used for exponential moving average. If heavy smoothing is applied (`lam`<<1), a larger `t_max` may be necessary in order to ensure the thresholds have converged.
* `n_features`: Number of features used in the FET test. No need to pass it if no preprocessing takes place. In case of a preprocessing step, this can also be inferred automatically but could be more expensive to compute.
* `verbose`: Whether or not to print progress during configuration.
* `input_shape`: Shape of input data.
* `data_type`: Optionally specify the data type (tabular, image or time-series). Added to metadata.

Initialized drift detector example:

```python
from alibi_detect.cd import FETDriftOnline

ert = 150
window_sizes = [20,40]
cd = FETDriftOnline(x_ref, ert, window_sizes)
```

### Detect Drift

We detect data drift by sequentially calling `predict` on single instances `x_t` (no batch dimension) as they each arrive. We can return the test-statistic and the threshold by setting `return_test_stat` to *True*.

The prediction takes the form of a dictionary with `meta` and `data` keys. `meta` contains the detector's metadata while `data` is also a dictionary which contains the actual predictions stored in the following keys:

* `is_drift`: 1 if any of the test-windows have drifted from the reference data and 0 otherwise.

* `time`: The number of observations that have been so far passed to the detector as test instances.

* `ert`: The expected run-time the detector was configured to run at in the absence of drift.

* `test_stat`: FET test-statistics (`1-p_val`) between the reference data and the test_windows if `return_test_stat` equals *True*.

* `threshold`: The values the test-statsitics are required to exceed for drift to be detected if `return_test_stat` equals *True*.


```python
preds = cd.predict(x_t, return_test_stat=True)
```

### Managing State

The detector's state may be saved with the `save_state` method:

```python
cd = FETDriftOnline(x_ref, ert, window_sizes)  # Instantiate detector at t=0
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

## References

[1] Ross, G.J., Tasoulis, D.K. & Adams, N.M. Sequential monitoring of a Bernoulli sequence when the pre-change parameter is unknown. Comput Stat 28, 463–479 (2013). doi: [10.1007/s00180-012-0311-7](https://doi.org/10.1007/s00180-012-0311-7). arXiv: [1212.6020](https://arxiv.org/abs/1212.6020).

