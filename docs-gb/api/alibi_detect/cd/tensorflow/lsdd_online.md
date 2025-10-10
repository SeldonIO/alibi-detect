# `alibi_detect.cd.tensorflow.lsdd_online`
## `LSDDDriftOnlineTF`

_Inherits from:_ `BaseMultiDriftOnline`, `BaseDetector`, `StateMixin`, `ABC`

### Constructor

```python
LSDDDriftOnlineTF(self, x_ref: Union[numpy.ndarray, list], ert: float, window_size: int, preprocess_fn: Optional[Callable] = None, x_ref_preprocessed: bool = False, sigma: Optional[numpy.ndarray] = None, n_bootstraps: int = 1000, n_kernel_centers: Optional[int] = None, lambda_rd_max: float = 0.2, verbose: bool = True, input_shape: Optional[tuple] = None, data_type: Optional[str] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `Union[numpy.ndarray, list]` |  | Data used as reference distribution. |
| `ert` | `float` |  | The expected run-time (ERT) in the absence of drift. For the multivariate detectors, the ERT is defined as the expected run-time from t=0. |
| `window_size` | `int` |  | The size of the sliding test-window used to compute the test-statistic. Smaller windows focus on responding quickly to severe drift, larger windows focus on ability to detect slight drift. |
| `preprocess_fn` | `Optional[Callable]` | `None` | Function to preprocess the data before computing the data drift metrics. |
| `x_ref_preprocessed` | `bool` | `False` | Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference data will also be preprocessed. |
| `sigma` | `Optional[numpy.ndarray]` | `None` | Optionally set the bandwidth of the Gaussian kernel used in estimating the LSDD. Can also pass multiple bandwidth values as an array. The kernel evaluation is then averaged over those bandwidths. If `sigma` is not specified, the 'median heuristic' is adopted whereby `sigma` is set as the median pairwise distance between reference samples. |
| `n_bootstraps` | `int` | `1000` | The number of bootstrap simulations used to configure the thresholds. The larger this is the more accurately the desired ERT will be targeted. Should ideally be at least an order of magnitude larger than the ert. |
| `n_kernel_centers` | `Optional[int]` | `None` | The number of reference samples to use as centers in the Gaussian kernel model used to estimate LSDD. Defaults to 2*window_size. |
| `lambda_rd_max` | `float` | `0.2` | The maximum relative difference between two estimates of LSDD that the regularization parameter lambda is allowed to cause. Defaults to 0.2 as in the paper. |
| `verbose` | `bool` | `True` | Whether or not to print progress during configuration. |
| `input_shape` | `Optional[tuple]` | `None` | Shape of input data. |
| `data_type` | `Optional[str]` | `None` | Optionally specify the data type (tabular, image or time-series). Added to metadata. |

### Methods

#### `score`

```python
score(x_t: Union[numpy.ndarray, typing.Any]) -> float
```

Compute the test-statistic (LSDD) between the reference window and test window.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_t` | `Union[numpy.ndarray, typing.Any]` |  | A single instance to be added to the test-window. |

**Returns**
- Type: `float`
