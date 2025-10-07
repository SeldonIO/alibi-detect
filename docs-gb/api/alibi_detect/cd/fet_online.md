# `alibi_detect.cd.fet_online`
## `FETDriftOnline`

_Inherits from:_ `BaseUniDriftOnline`, `BaseDetector`, `StateMixin`, `ABC`, `DriftConfigMixin`

### Constructor

```python
FETDriftOnline(self, x_ref: Union[numpy.ndarray, list], ert: float, window_sizes: List[int], preprocess_fn: Optional[Callable] = None, x_ref_preprocessed: bool = False, n_bootstraps: int = 10000, t_max: Optional[int] = None, alternative: str = 'greater', lam: float = 0.99, n_features: Optional[int] = None, verbose: bool = True, input_shape: Optional[tuple] = None, data_type: Optional[str] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `Union[numpy.ndarray, list]` |  | Data used as reference distribution. |
| `ert` | `float` |  | The expected run-time (ERT) in the absence of drift. For the univariate detectors, the ERT is defined as the expected run-time after the smallest window is full i.e. the run-time from t=min(windows_sizes). |
| `window_sizes` | `List[int]` |  | window sizes for the sliding test-windows used to compute the test-statistic. Smaller windows focus on responding quickly to severe drift, larger windows focus on ability to detect slight drift. |
| `preprocess_fn` | `Optional[Callable]` | `None` | Function to preprocess the data before computing the data drift metrics. |
| `x_ref_preprocessed` | `bool` | `False` | Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference data will also be preprocessed. |
| `n_bootstraps` | `int` | `10000` | The number of bootstrap simulations used to configure the thresholds. The larger this is the more accurately the desired ERT will be targeted. Should ideally be at least an order of magnitude larger than the ERT. |
| `t_max` | `Optional[int]` | `None` | Length of the streams to simulate when configuring thresholds. If `None`, this is set to 2 * max(`window_sizes`) - 1. |
| `alternative` | `str` | `'greater'` | Defines the alternative hypothesis. Options are 'greater' or 'less', which correspond to an increase or decrease in the mean of the Bernoulli stream. |
| `lam` | `float` | `0.99` | Smoothing coefficient used for exponential moving average. |
| `n_features` | `Optional[int]` | `None` | Number of features used in the statistical test. No need to pass it if no preprocessing takes place. In case of a preprocessing step, this can also be inferred automatically but could be more expensive to compute. |
| `verbose` | `bool` | `True` | Whether or not to print progress during configuration. |
| `input_shape` | `Optional[tuple]` | `None` | Shape of input data. |
| `data_type` | `Optional[str]` | `None` | Optionally specify the data type (tabular, image or time-series). Added to metadata. |

### Methods

#### `score`

```python
score(x_t: Union[numpy.ndarray, typing.Any]) -> numpy.ndarray
```

Compute the test-statistic (FET) between the reference window(s) and test window.

If a given test-window is not yet full then a test-statistic of np.nan is returned for that window.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_t` | `Union[numpy.ndarray, typing.Any]` |  | A single instance. |

**Returns**
- Type: `numpy.ndarray`
