# `alibi_detect.od.sr`
## Constants
### `logger`
```python
logger: logging.Logger = <Logger alibi_detect.od.sr (WARNING)>
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

### `EPSILON`
```python
EPSILON: float = 1e-08
```
Convert a string or number to a floating point number, if possible.

## `Padding`

_Inherits from:_ `str`, `Enum`

An enumeration.

## `Side`

_Inherits from:_ `str`, `Enum`

An enumeration.

## `SpectralResidual`

_Inherits from:_ `BaseDetector`, `ThresholdMixin`, `ABC`

### Constructor

```python
SpectralResidual(self, threshold: float = None, window_amp: int = None, window_local: int = None, padding_amp_method: typing_extensions.Literal['constant', 'replicate', 'reflect'] = 'reflect', padding_local_method: typing_extensions.Literal['constant', 'replicate', 'reflect'] = 'reflect', padding_amp_side: typing_extensions.Literal['bilateral', 'left', 'right'] = 'bilateral', n_est_points: int = None, n_grad_points: int = 5) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `threshold` | `Optional[float]` | `None` | Threshold used to classify outliers. Relative saliency map distance from the moving average. |
| `window_amp` | `Optional[int]` | `None` | Window for the average log amplitude. |
| `window_local` | `Optional[int]` | `None` | Window for the local average of the saliency map. Note that the averaging is performed over the previous `window_local` data points (i.e., is a local average of the preceding `window_local` points for the current index). |
| `padding_amp_method` | `Literal[constant, replicate, reflect]` | `'reflect'` | Padding method to be used prior to each convolution over log amplitude. Possible values: `constant` | `replicate` | `reflect`. Default value: `replicate`. |
| `padding_local_method` | `Literal[constant, replicate, reflect]` | `'reflect'` | Padding method to be used prior to each convolution over saliency map. Possible values: `constant` | `replicate` | `reflect`. Default value: `replicate`. |
| `padding_amp_side` | `Literal[bilateral, left, right]` | `'bilateral'` | Whether to pad the amplitudes on both sides or only on one side. Possible values: `bilateral` | `left` | `right`. |
| `n_est_points` | `Optional[int]` | `None` | Number of estimated points padded to the end of the sequence. |
| `n_grad_points` | `int` | `5` | Number of points used for the gradient estimation of the additional points padded to the end of the sequence. |

### Methods

#### `add_est_points`

```python
add_est_points(X: numpy.ndarray, t: numpy.ndarray) -> numpy.ndarray
```

Pad the time series with additional points since the method works better if the anomaly point

is towards the center of the sliding window.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Uniformly sampled time series instances. |
| `t` | `numpy.ndarray` |  | Equidistant timestamps corresponding to each input instances (i.e, the array should contain numerical values in increasing order). |

**Returns**
- Type: `numpy.ndarray`

#### `compute_grads`

```python
compute_grads(X: numpy.ndarray, t: numpy.ndarray) -> numpy.ndarray
```

Slope of the straight line between different points of the time series

multiplied by the average time step size.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Uniformly sampled time series instances. |
| `t` | `numpy.ndarray` |  | Equidistant timestamps corresponding to each input instances (i.e, the array should contain numerical values in increasing order). |

**Returns**
- Type: `numpy.ndarray`

#### `infer_threshold`

```python
infer_threshold(X: numpy.ndarray, t: Optional[numpy.ndarray] = None, threshold_perc: float = 95.0) -> None
```

Update threshold by a value inferred from the percentage of instances considered to be

outliers in a sample of the dataset.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Uniformly sampled time series instances. |
| `t` | `Optional[numpy.ndarray]` | `None` | Equidistant timestamps corresponding to each input instances (i.e, the array should contain numerical values in increasing order). If not provided, the timestamps will be replaced by an array of integers `[0, 1, ... , N - 1]`, where `N` is the size of the input time series. |
| `threshold_perc` | `float` | `95.0` | Percentage of `X` considered to be normal based on the outlier score. |

**Returns**
- Type: `None`

#### `pad_same`

```python
pad_same(X: numpy.ndarray, W: numpy.ndarray, method: str = 'replicate', side: str = 'bilateral') -> numpy.ndarray
```

Adds padding to the time series `X` such that after applying a valid convolution with a kernel/filter

`w`, the resulting time series has the same shape as the input `X`.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Time series to be padded |
| `W` | `numpy.ndarray` |  | Convolution kernel/filter. |
| `method` | `str` | `'replicate'` | Padding method to be used. Possible values: |
| `side` | `str` | `'bilateral'` | Whether to pad the time series bilateral or only on one side. Possible values: |

**Returns**
- Type: `numpy.ndarray`

#### `predict`

```python
predict(X: numpy.ndarray, t: Optional[numpy.ndarray] = None, return_instance_score: bool = True) -> Dict[Dict[str, str], Dict[numpy.ndarray, numpy.ndarray]]
```

Compute outlier scores and transform into outlier predictions.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Uniformly sampled time series instances. |
| `t` | `Optional[numpy.ndarray]` | `None` | Equidistant timestamps corresponding to each input instances (i.e, the array should contain numerical values in increasing order). If not provided, the timestamps will be replaced by an array of integers `[0, 1, ... , N - 1]`, where `N` is the size of the input time series. |
| `return_instance_score` | `bool` | `True` | Whether to return instance level outlier scores. |

**Returns**
- Type: `Dict[Dict[str, str], Dict[numpy.ndarray, numpy.ndarray]]`

#### `saliency_map`

```python
saliency_map(X: numpy.ndarray) -> numpy.ndarray
```

Compute saliency map.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Uniformly sampled time series instances. |

**Returns**
- Type: `numpy.ndarray`

#### `score`

```python
score(X: numpy.ndarray, t: Optional[numpy.ndarray] = None) -> numpy.ndarray
```

Compute outlier scores.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Uniformly sampled time series instances. |
| `t` | `Optional[numpy.ndarray]` | `None` | Equidistant timestamps corresponding to each input instances (i.e, the array should contain numerical values in increasing order). If not provided, the timestamps will be replaced by an array of integers `[0, 1, ... , N - 1]`, where `N` is the size of the input time series. |

**Returns**
- Type: `numpy.ndarray`
