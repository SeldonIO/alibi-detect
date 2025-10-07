# `alibi_detect.od.prophet`
## Constants
### `logger`
```python
logger: logging.Logger = <Logger alibi_detect.od.prophet (WARNING)>
```
Instances of the Logger class represent a single logging channel. A
"logging channel" indicates an area of an application. Exactly how an
"area" is defined is up to the application developer. Since an
application can have any number of areas, logging channels are identified
by a unique string. Application areas can be nested (e.g. an area
of "input processing" might include sub-areas "read CSV files", "read
XLS files" and "read Gnumeric files"). To cater for this natural nesting,
channel names are organized into a namespace hierarchy where levels are
separated by periods, much like the Java or Python package namespace. So
in the instance given above, channel names might be "input" for the upper
level, and "input.csv", "input.xls" and "input.gnu" for the sub-levels.
There is no arbitrary limit to the depth of nesting.

## `OutlierProphet`

_Inherits from:_ `BaseDetector`, `FitMixin`, `ABC`

### Constructor

```python
OutlierProphet(self, threshold: float = 0.8, growth: str = 'linear', cap: float = None, holidays: pandas.core.frame.DataFrame = None, holidays_prior_scale: float = 10.0, country_holidays: str = None, changepoint_prior_scale: float = 0.05, changepoint_range: float = 0.8, seasonality_mode: str = 'additive', daily_seasonality: Union[str, bool, int] = 'auto', weekly_seasonality: Union[str, bool, int] = 'auto', yearly_seasonality: Union[str, bool, int] = 'auto', add_seasonality: List = None, seasonality_prior_scale: float = 10.0, uncertainty_samples: int = 1000, mcmc_samples: int = 0) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `threshold` | `float` | `0.8` | Width of the uncertainty intervals of the forecast, used as outlier threshold. Equivalent to `interval_width`. If the instance lies outside of the uncertainty intervals, it is flagged as an outlier. If `mcmc_samples` equals 0, it is the uncertainty in the trend using the MAP estimate of the extrapolated model. If `mcmc_samples` >0, then uncertainty over all parameters is used. |
| `growth` | `str` | `'linear'` | 'linear' or 'logistic' to specify a linear or logistic trend. |
| `cap` | `Optional[float]` | `None` | Growth cap in case growth equals 'logistic'. |
| `holidays` | `Optional[pandas.core.frame.DataFrame]` | `None` | pandas DataFrame with columns `holiday` (string) and `ds` (dates) and optionally columns `lower_window` and `upper_window` which specify a range of days around the date to be included as holidays. |
| `holidays_prior_scale` | `float` | `10.0` | Parameter controlling the strength of the holiday components model. Higher values imply a more flexible trend, more prone to more overfitting. |
| `country_holidays` | `Optional[str]` | `None` | Include country-specific holidays via country abbreviations. The holidays for each country are provided by the holidays package in Python. A list of available countries and the country name to use is available on: https://github.com/dr-prodigy/python-holidays. Additionally, Prophet includes holidays for: Brazil (BR), Indonesia (ID), India (IN), Malaysia (MY), Vietnam (VN), Thailand (TH), Philippines (PH), Turkey (TU), Pakistan (PK), Bangladesh (BD), Egypt (EG), China (CN) and Russian (RU). |
| `changepoint_prior_scale` | `float` | `0.05` | Parameter controlling the flexibility of the automatic changepoint selection. Large values will allow many changepoints, potentially leading to overfitting. |
| `changepoint_range` | `float` | `0.8` | Proportion of history in which trend changepoints will be estimated. Higher values means more changepoints, potentially leading to overfitting. |
| `seasonality_mode` | `str` | `'additive'` | Either 'additive' or 'multiplicative'. |
| `daily_seasonality` | `Union[str, bool, int]` | `'auto'` | Can be 'auto', True, False, or a number of Fourier terms to generate. |
| `weekly_seasonality` | `Union[str, bool, int]` | `'auto'` | Can be 'auto', True, False, or a number of Fourier terms to generate. |
| `yearly_seasonality` | `Union[str, bool, int]` | `'auto'` | Can be 'auto', True, False, or a number of Fourier terms to generate. |
| `add_seasonality` | `Optional[List[Any]]` | `None` | Manually add one or more seasonality components. Pass a list of dicts containing the keys `name`, `period`, `fourier_order` (obligatory), `prior_scale` and `mode` (optional). |
| `seasonality_prior_scale` | `float` | `10.0` | Parameter controlling the strength of the seasonality model. Larger values allow the model to fit larger seasonal fluctuations, potentially leading to overfitting. |
| `uncertainty_samples` | `int` | `1000` | Number of simulated draws used to estimate uncertainty intervals. |
| `mcmc_samples` | `int` | `0` | If >0, will do full Bayesian inference with the specified number of MCMC samples. If 0, will do MAP estimation. |

### Methods

#### `fit`

```python
fit(df: pandas.core.frame.DataFrame) -> None
```

Fit Prophet model on normal (inlier) data.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `df` | `pandas.core.frame.DataFrame` |  | Dataframe with columns `ds` with timestamps and `y` with target values. |

**Returns**
- Type: `None`

#### `predict`

```python
predict(df: pandas.core.frame.DataFrame, return_instance_score: bool = True, return_forecast: bool = True) -> Dict[Dict[str, str], Dict[pandas.core.frame.DataFrame, pandas.core.frame.DataFrame]]
```

Compute outlier scores and transform into outlier predictions.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `df` | `pandas.core.frame.DataFrame` |  | DataFrame with columns `ds` with timestamps and `y` with values which need to be flagged as outlier or not. |
| `return_instance_score` | `bool` | `True` | Whether to return instance level outlier scores. |
| `return_forecast` | `bool` | `True` | Whether to return the model forecast. |

**Returns**
- Type: `Dict[Dict[str, str], Dict[pandas.core.frame.DataFrame, pandas.core.frame.DataFrame]]`

#### `score`

```python
score(df: pandas.core.frame.DataFrame) -> pandas.core.frame.DataFrame
```

Compute outlier scores.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `df` | `pandas.core.frame.DataFrame` |  | DataFrame with columns `ds` with timestamps and `y` with values which need to be flagged as outlier or not. |

**Returns**
- Type: `pandas.core.frame.DataFrame`
