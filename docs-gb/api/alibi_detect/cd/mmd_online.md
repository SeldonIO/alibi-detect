# `alibi_detect.cd.mmd_online`
## Constants
### `has_pytorch`
```python
has_pytorch: bool = True
```
bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### `has_tensorflow`
```python
has_tensorflow: bool = True
```
bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

## `MMDDriftOnline`

_Inherits from:_ `DriftConfigMixin`

### Constructor

```python
MMDDriftOnline(self, x_ref: Union[numpy.ndarray, list], ert: float, window_size: int, backend: str = 'tensorflow', preprocess_fn: Optional[Callable] = None, x_ref_preprocessed: bool = False, kernel: Optional[Callable] = None, sigma: Optional[numpy.ndarray] = None, n_bootstraps: int = 1000, device: Union[typing_extensions.Literal['cuda', 'gpu', 'cpu'], ForwardRef('torch.device'), NoneType] = None, verbose: bool = True, input_shape: Optional[tuple] = None, data_type: Optional[str] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `Union[numpy.ndarray, list]` |  | Data used as reference distribution. |
| `ert` | `float` |  | The expected run-time (ERT) in the absence of drift. For the multivariate detectors, the ERT is defined as the expected run-time from t=0. |
| `window_size` | `int` |  | The size of the sliding test-window used to compute the test-statistic. Smaller windows focus on responding quickly to severe drift, larger windows focus on ability to detect slight drift. |
| `backend` | `str` | `'tensorflow'` | Backend used for the MMD implementation and configuration. |
| `preprocess_fn` | `Optional[Callable]` | `None` | Function to preprocess the data before computing the data drift metrics. |
| `x_ref_preprocessed` | `bool` | `False` | Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference data will also be preprocessed. |
| `kernel` | `Optional[Callable]` | `None` | Kernel used for the MMD computation, defaults to Gaussian RBF kernel. |
| `sigma` | `Optional[numpy.ndarray]` | `None` | Optionally set the GaussianRBF kernel bandwidth. Can also pass multiple bandwidth values as an array. The kernel evaluation is then averaged over those bandwidths. If `sigma` is not specified, the 'median heuristic' is adopted whereby `sigma` is set as the median pairwise distance between reference samples. |
| `n_bootstraps` | `int` | `1000` | The number of bootstrap simulations used to configure the thresholds. The larger this is the more accurately the desired ERT will be targeted. Should ideally be at least an order of magnitude larger than the ERT. |
| `device` | `Union[Literal[cuda, gpu, cpu], ForwardRef('torch.device'), None]` | `None` | Device type used. The default tries to use the GPU and falls back on CPU if needed. Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of ``torch.device``. Only relevant for 'pytorch' backend. |
| `verbose` | `bool` | `True` | Whether or not to print progress during configuration. |
| `input_shape` | `Optional[tuple]` | `None` | Shape of input data. |
| `data_type` | `Optional[str]` | `None` | Optionally specify the data type (tabular, image or time-series). Added to metadata. |

### Properties

| Property | Type | Description |
| -------- | ---- | ----------- |
| `t` | `` |  |
| `test_stats` | `` |  |
| `thresholds` | `` |  |

### Methods

#### `get_config`

```python
get_config() -> dict
```

**Returns**
- Type: `dict`

#### `load_state`

```python
load_state(filepath: Union[str, os.PathLike])
```

Load the detector's state from disk, in order to restart from a checkpoint previously generated with

`save_state`.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `filepath` | `Union[str, os.PathLike]` |  | The directory to load state from. |

#### `predict`

```python
predict(x_t: Union[numpy.ndarray, typing.Any], return_test_stat: bool = True) -> Dict[Dict[str, str], Dict[str, Union[int, float]]]
```

Predict whether the most recent window of data has drifted from the reference data.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_t` | `Union[numpy.ndarray, typing.Any]` |  | A single instance to be added to the test-window. |
| `return_test_stat` | `bool` | `True` | Whether to return the test statistic (squared MMD) and threshold. |

**Returns**
- Type: `Dict[Dict[str, str], Dict[str, Union[int, float]]]`

#### `reset_state`

```python
reset_state()
```

Resets the detector to its initial state (`t=0`). This does not include reconfiguring thresholds.

#### `save_state`

```python
save_state(filepath: Union[str, os.PathLike])
```

Save a detector's state to disk in order to generate a checkpoint.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `filepath` | `Union[str, os.PathLike]` |  | The directory to save state to. |

#### `score`

```python
score(x_t: Union[numpy.ndarray, typing.Any]) -> float
```

Compute the test-statistic (squared MMD) between the reference window and test window.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_t` | `Union[numpy.ndarray, typing.Any]` |  | A single instance to be added to the test-window. |

**Returns**
- Type: `float`
