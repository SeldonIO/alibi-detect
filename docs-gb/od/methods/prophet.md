---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

[source](../../api/alibi_detect.od.prophet.rst)

# Prophet Detector

## Overview

The Prophet outlier detector uses the [Prophet](https://facebook.github.io/prophet/) time series forecasting package explained in [this excellent paper](https://peerj.com/preprints/3190/). The underlying Prophet model is a decomposable univariate time series model combining trend, seasonality and holiday effects. The model forecast also includes an uncertainty interval around the estimated trend component using the [MAP estimate](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation) of the extrapolated model. Alternatively, full Bayesian inference can be done at the expense of increased compute. The upper and lower values of the uncertainty interval can then be used as outlier thresholds for each point in time. First, the distance from the observed value to the nearest uncertainty boundary (upper or lower) is computed. If the observation is within the boundaries, the outlier score equals the negative distance. As a result, the outlier score is the lowest when the observation equals the model prediction. If the observation is outside of the boundaries, the score equals the distance measure and the observation is flagged as an outlier. One of the main drawbacks of the method however is that you need to refit the model as new data comes in. This is undesirable for applications with high throughput and real-time detection.

<div class="alert alert-info">
Note

To use this detector, first install Prophet by running:    
```bash
pip install alibi-detect[prophet]
```

This will install Prophet, and its major dependency PyStan. PyStan is currently only [partly supported on Windows](https://pystan.readthedocs.io/en/stable/faq.html?highlight=windows#is-windows-supported). If this detector is to be used on a Windows system, it is recommended to manually install (and test) PyStan before running the command above.

</div>

## Usage

### Initialize

Parameters:

* `threshold`: width of the uncertainty intervals of the forecast, used as outlier threshold. Equivalent to `interval_width`. If the instance lies outside of the uncertainty intervals, it is flagged as an outlier. If `mcmc_samples` equals 0, it is the uncertainty in the trend using the MAP estimate of the extrapolated model. If `mcmc_samples` >0, then uncertainty over all parameters is used.

* `growth`: *'linear'* or *'logistic'* to specify a linear or logistic trend.

* `cap`: growth cap in case growth equals *'logistic'*.

* `holidays`: pandas DataFrame with columns *'holiday'* (string) and *'ds'* (dates) and optionally columns *'lower_window'* and *'upper_window'* which specify a range of days around the date to be included as holidays.

* `holidays_prior_scale`: parameter controlling the strength of the holiday components model. Higher values imply a more flexible trend, more prone to more overfitting.

* `country_holidays`: include country-specific holidays via country abbreviations. The holidays for each country are provided by the holidays package in Python. A list of available countries and the country name to use is available on: https://github.com/dr-prodigy/python-holidays. Additionally, Prophet includes holidays for: Brazil (BR), Indonesia (ID), India (IN), Malaysia (MY), Vietnam (VN), Thailand (TH), Philippines (PH), Turkey (TU), Pakistan (PK), Bangladesh (BD), Egypt (EG), China (CN) and Russian (RU).

* `changepoint_prior_scale`: parameter controlling the flexibility of the automatic changepoint selection. Large values will allow many changepoints, potentially leading to overfitting.

* `changepoint_range`: proportion of history in which trend changepoints will be estimated. Higher values means more changepoints, potentially leading to overfitting.

* `seasonality_mode`: either *'additive'* or *'multiplicative'*.

* `daily_seasonality`: can be *'auto'*, True, False, or a number of Fourier terms to generate.

* `weekly_seasonality`: can be *'auto'*, True, False, or a number of Fourier terms to generate.

* `yearly_seasonality`: can be *'auto'*, True, False, or a number of Fourier terms to generate.

* `add_seasonality`: manually add one or more seasonality components. Pass a list of dicts containing the keys *'name'*, *'period'*, *'fourier_order'* (obligatory), *'prior_scale'* and *'mode'* (optional).

* `seasonality_prior_scale`: parameter controlling the strength of the seasonality model. Larger values allow the model to fit larger seasonal fluctuations, potentially leading to overfitting.

* `uncertainty_samples`: number of simulated draws used to estimate uncertainty intervals.

* `mcmc_samples`: If *> 0*, will do full Bayesian inference with the specified number of MCMC samples. If *0*, will do MAP estimation.



Initialized outlier detector example:

```python
from alibi_detect.od import OutlierProphet

od = OutlierProphet(
    threshold=0.9,
    growth='linear'
)
```

### Fit

We then need to train the outlier detector. The `fit` method takes a pandas DataFrame *df* with as columns *'ds'* containing the dates or timestamps and *'y'* for the time series being investigated. The date format is ideally *YYYY-MM-DD* and timestamp format *YYYY-MM-DD HH:MM:SS*.

```python
od.fit(df)
```

### Detect

We detect outliers by simply calling `predict` on a DataFrame *df*, again with columns *'ds'* and *'y'* to compute the instance level outlier scores. We can also return the instance level outlier score or the raw Prophet model forecast by setting respectively `return_instance_score` or `return_forecast` to True. **It is important that the dates or timestamps of the test data follow the training data**.

The prediction takes the form of a dictionary with `meta` and `data` keys. `meta` contains the detector's metadata while `data` is also a dictionary which contains the actual predictions stored in the following keys:

* `is_outlier`: DataFrame with columns *'ds'* containing the dates or timestamps and *'is_outlier'* a boolean whether instances are above the threshold and therefore outlier instances.

* `instance_score`: DataFrame with *'ds'* and *'instance_score'* which contains instance level scores if `return_instance_score` equals True.

* `forecast`: DataFrame with the raw model predictions if `return_forecast` equals True. The DataFrame contains columns with the upper and lower boundaries (*'yhat_upper'* and *'yhat_lower'*), the model predictions (*'yhat'*), and the decomposition of the prediction in the different components (trend, seasonality, holiday). 


```python
preds = od.predict(
    df,
    return_instance_score=True,
    return_forecast=True
)
```

## Examples

[Time-series outlier detection using Prophet on weather data](../../examples/od_prophet_weather.ipynb)

