# `alibi_detect.cd.lsdd`
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

## `LSDDDrift`

_Inherits from:_ `DriftConfigMixin`

### Constructor

```python
LSDDDrift(self, x_ref: Union[numpy.ndarray, list], backend: str = 'tensorflow', p_val: float = 0.05, x_ref_preprocessed: bool = False, preprocess_at_init: bool = True, update_x_ref: Optional[Dict[str, int]] = None, preprocess_fn: Optional[Callable] = None, sigma: Optional[numpy.ndarray] = None, n_permutations: int = 100, n_kernel_centers: Optional[int] = None, lambda_rd_max: float = 0.2, device: Union[typing_extensions.Literal['cuda', 'gpu', 'cpu'], ForwardRef('torch.device'), NoneType] = None, input_shape: Optional[tuple] = None, data_type: Optional[str] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `Union[numpy.ndarray, list]` |  | Data used as reference distribution. |
| `backend` | `str` | `'tensorflow'` | Backend used for the LSDD implementation. |
| `p_val` | `float` | `0.05` | p-value used for the significance of the permutation test. |
| `x_ref_preprocessed` | `bool` | `False` | Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference data will also be preprocessed. |
| `preprocess_at_init` | `bool` | `True` | Whether to preprocess the reference data when the detector is instantiated. Otherwise, the reference data will be preprocessed at prediction time. Only applies if `x_ref_preprocessed=False`. |
| `update_x_ref` | `Optional[Dict[str, int]]` | `None` | Reference data can optionally be updated to the last n instances seen by the detector or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while for reservoir sampling {'reservoir_sampling': n} is passed. |
| `preprocess_fn` | `Optional[Callable]` | `None` | Function to preprocess the data before computing the data drift metrics. |
| `sigma` | `Optional[numpy.ndarray]` | `None` | Optionally set the bandwidth of the Gaussian kernel used in estimating the LSDD. Can also pass multiple bandwidth values as an array. The kernel evaluation is then averaged over those bandwidths. If `sigma` is not specified, the 'median heuristic' is adopted whereby `sigma` is set as the median pairwise distance between reference samples. |
| `n_permutations` | `int` | `100` | Number of permutations used in the permutation test. |
| `n_kernel_centers` | `Optional[int]` | `None` | The number of reference samples to use as centers in the Gaussian kernel model used to estimate LSDD. Defaults to 1/20th of the reference data. |
| `lambda_rd_max` | `float` | `0.2` | The maximum relative difference between two estimates of LSDD that the regularization parameter lambda is allowed to cause. Defaults to 0.2 as in the paper. |
| `device` | `Union[Literal[cuda, gpu, cpu], ForwardRef('torch.device'), None]` | `None` | Device type used. The default tries to use the GPU and falls back on CPU if needed. Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of ``torch.device``. Only relevant for 'pytorch' backend. |
| `input_shape` | `Optional[tuple]` | `None` | Shape of input data. |
| `data_type` | `Optional[str]` | `None` | Optionally specify the data type (tabular, image or time-series). Added to metadata. |

### Methods

#### `get_config`

```python
get_config() -> dict
```

**Returns**
- Type: `dict`

#### `predict`

```python
predict(x: Union[numpy.ndarray, list], return_p_val: bool = True, return_distance: bool = True) -> Dict[Dict[str, str], Dict[str, Union[int, float]]]
```

Predict whether a batch of data has drifted from the reference data.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, list]` |  | Batch of instances. |
| `return_p_val` | `bool` | `True` | Whether to return the p-value of the permutation test. |
| `return_distance` | `bool` | `True` | Whether to return the LSDD metric between the new batch and reference data. |

**Returns**
- Type: `Dict[Dict[str, str], Dict[str, Union[int, float]]]`

#### `score`

```python
score(x: Union[numpy.ndarray, list]) -> Tuple[float, float, float]
```

Compute the p-value resulting from a permutation test using the least-squares density

difference as a distance measure between the reference data and the data to be tested.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, list]` |  | Batch of instances. |

**Returns**
- Type: `Tuple[float, float, float]`
