# `alibi_detect.cd.base_online`
## Constants
### `TYPE_CHECKING`
```python
TYPE_CHECKING: bool = False
```
bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### `logger`
```python
logger: logging.Logger = <Logger alibi_detect.cd.base_online (WARNING)>
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

## `BaseMultiDriftOnline`

_Inherits from:_ `BaseDetector`, `StateMixin`, `ABC`

### Constructor

```python
BaseMultiDriftOnline(self, x_ref: Union[numpy.ndarray, list], ert: float, window_size: int, preprocess_fn: Optional[Callable] = None, x_ref_preprocessed: bool = False, n_bootstraps: int = 1000, verbose: bool = True, input_shape: Optional[tuple] = None, data_type: Optional[str] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `Union[numpy.ndarray, list]` |  | Data used as reference distribution. |
| `ert` | `float` |  | The expected run-time (ERT) in the absence of drift. For the multivariate detectors, the ERT is defined as the expected run-time from t=0. |
| `window_size` | `int` |  | The size of the sliding test-window used to compute the test-statistic. Smaller windows focus on responding quickly to severe drift, larger windows focus on ability to detect slight drift. |
| `preprocess_fn` | `Optional[Callable]` | `None` | Function to preprocess the data before computing the data drift metrics. |
| `x_ref_preprocessed` | `bool` | `False` | Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference data will also be preprocessed. |
| `n_bootstraps` | `int` | `1000` | The number of bootstrap simulations used to configure the thresholds. The larger this is the more accurately the desired ERT will be targeted. Should ideally be at least an order of magnitude larger than the ert. |
| `verbose` | `bool` | `True` | Whether or not to print progress during configuration. |
| `input_shape` | `Optional[tuple]` | `None` | Shape of input data. |
| `data_type` | `Optional[str]` | `None` | Optionally specify the data type (tabular, image or time-series). Added to metadata. |

### Methods

#### `get_threshold`

```python
get_threshold(t: int) -> float
```

Return the threshold for timestep `t`.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `t` | `int` |  | The timestep to return a threshold for. |

**Returns**
- Type: `float`

#### `predict`

```python
predict(x_t: Union[numpy.ndarray, typing.Any], return_test_stat: bool = True) -> Dict[Dict[str, str], Dict[str, Union[int, float]]]
```

Predict whether the most recent window of data has drifted from the reference data.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_t` | `Union[numpy.ndarray, typing.Any]` |  | A single instance to be added to the test-window. |
| `return_test_stat` | `bool` | `True` | Whether to return the test statistic and threshold. |

**Returns**
- Type: `Dict[Dict[str, str], Dict[str, Union[int, float]]]`

#### `reset`

```python
reset() -> None
```

Deprecated reset method. This method will be repurposed or removed in the future. To reset the detector to

its initial state (`t=0`) use :meth:`reset_state`.

**Returns**
- Type: `None`

#### `reset_state`

```python
reset_state() -> None
```

Resets the detector to its initial state (`t=0`). This does not include reconfiguring thresholds.

**Returns**
- Type: `None`

## `BaseUniDriftOnline`

_Inherits from:_ `BaseDetector`, `StateMixin`, `ABC`

### Constructor

```python
BaseUniDriftOnline(self, x_ref: Union[numpy.ndarray, list], ert: float, window_sizes: List[int], preprocess_fn: Optional[Callable] = None, x_ref_preprocessed: bool = False, n_bootstraps: int = 1000, n_features: Optional[int] = None, verbose: bool = True, input_shape: Optional[tuple] = None, data_type: Optional[str] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `Union[numpy.ndarray, list]` |  | Data used as reference distribution. |
| `ert` | `float` |  | The expected run-time (ERT) in the absence of drift. For the univariate detectors, the ERT is defined as the expected run-time after the smallest window is full i.e. the run-time from t=min(windows_sizes)-1. |
| `window_sizes` | `List[int]` |  | The sizes of the sliding test-windows used to compute the test-statistic. Smaller windows focus on responding quickly to severe drift, larger windows focus on ability to detect slight drift. |
| `preprocess_fn` | `Optional[Callable]` | `None` | Function to preprocess the data before computing the data drift metrics. |
| `x_ref_preprocessed` | `bool` | `False` | Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference data will also be preprocessed. |
| `n_bootstraps` | `int` | `1000` | The number of bootstrap simulations used to configure the thresholds. The larger this is the more accurately the desired ERT will be targeted. Should ideally be at least an order of magnitude larger than the ert. |
| `n_features` | `Optional[int]` | `None` | Number of features used in the statistical test. No need to pass it if no preprocessing takes place. In case of a preprocessing step, this can also be inferred automatically but could be more expensive to compute. |
| `verbose` | `bool` | `True` | Whether or not to print progress during configuration. |
| `input_shape` | `Optional[tuple]` | `None` | Shape of input data. |
| `data_type` | `Optional[str]` | `None` | Optionally specify the data type (tabular, image or time-series). Added to metadata. |

### Methods

#### `get_threshold`

```python
get_threshold(t: int) -> numpy.ndarray
```

Return the threshold for timestep `t`.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `t` | `int` |  | The timestep to return a threshold for. |

**Returns**
- Type: `numpy.ndarray`

#### `predict`

```python
predict(x_t: Union[numpy.ndarray, typing.Any], return_test_stat: bool = True) -> Dict[Dict[str, str], Dict[str, Union[int, float]]]
```

Predict whether the most recent window(s) of data have drifted from the reference data.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_t` | `Union[numpy.ndarray, typing.Any]` |  | A single instance to be added to the test-window(s). |
| `return_test_stat` | `bool` | `True` | Whether to return the test statistic and threshold. |

**Returns**
- Type: `Dict[Dict[str, str], Dict[str, Union[int, float]]]`

#### `reset`

```python
reset() -> None
```

Deprecated reset method. This method will be repurposed or removed in the future. To reset the detector to

its initial state (`t=0`) use :meth:`reset_state`.

**Returns**
- Type: `None`

#### `reset_state`

```python
reset_state() -> None
```

Resets the detector to its initial state (`t=0`). This does not include reconfiguring thresholds.

**Returns**
- Type: `None`
