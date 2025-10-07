# `alibi_detect.cd.mmd`
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

### `has_keops`
```python
has_keops: bool = True
```
bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### `logger`
```python
logger: logging.Logger = <Logger alibi_detect.cd.mmd (WARNING)>
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

## `MMDDrift`

_Inherits from:_ `DriftConfigMixin`

### Constructor

```python
MMDDrift(self, x_ref: Union[numpy.ndarray, list], backend: str = 'tensorflow', p_val: float = 0.05, x_ref_preprocessed: bool = False, preprocess_at_init: bool = True, update_x_ref: Optional[Dict[str, int]] = None, preprocess_fn: Optional[Callable] = None, kernel: Callable = None, sigma: Optional[numpy.ndarray] = None, configure_kernel_from_x_ref: bool = True, n_permutations: int = 100, batch_size_permutations: int = 1000000, device: Union[typing_extensions.Literal['cuda', 'gpu', 'cpu'], ForwardRef('torch.device'), NoneType] = None, input_shape: Optional[tuple] = None, data_type: Optional[str] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `Union[numpy.ndarray, list]` |  | Data used as reference distribution. |
| `backend` | `str` | `'tensorflow'` | Backend used for the MMD implementation. |
| `p_val` | `float` | `0.05` | p-value used for the significance of the permutation test. |
| `x_ref_preprocessed` | `bool` | `False` | Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference data will also be preprocessed. |
| `preprocess_at_init` | `bool` | `True` | Whether to preprocess the reference data when the detector is instantiated. Otherwise, the reference data will be preprocessed at prediction time. Only applies if `x_ref_preprocessed=False`. |
| `update_x_ref` | `Optional[Dict[str, int]]` | `None` | Reference data can optionally be updated to the last n instances seen by the detector or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while for reservoir sampling {'reservoir_sampling': n} is passed. |
| `preprocess_fn` | `Optional[Callable]` | `None` | Function to preprocess the data before computing the data drift metrics. |
| `kernel` | `Callable` | `None` | Kernel used for the MMD computation, defaults to Gaussian RBF kernel. |
| `sigma` | `Optional[numpy.ndarray]` | `None` | Optionally set the GaussianRBF kernel bandwidth. Can also pass multiple bandwidth values as an array. The kernel evaluation is then averaged over those bandwidths. |
| `configure_kernel_from_x_ref` | `bool` | `True` | Whether to already configure the kernel bandwidth from the reference data. |
| `n_permutations` | `int` | `100` | Number of permutations used in the permutation test. |
| `batch_size_permutations` | `int` | `1000000` | KeOps computes the n_permutations of the MMD^2 statistics in chunks of batch_size_permutations. Only relevant for 'keops' backend. |
| `device` | `Union[Literal[cuda, gpu, cpu], ForwardRef('torch.device'), None]` | `None` | Device type used. The default tries to use the GPU and falls back on CPU if needed. Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of ``torch.device``. Only relevant for 'pytorch' backend. |
| `input_shape` | `Optional[tuple]` | `None` | Shape of input data. |
| `data_type` | `Optional[str]` | `None` | Optionally specify the data type (tabular, image or time-series). Added to metadata. |

### Methods

#### `predict`

```python
predict(x: Union[numpy.ndarray, list], return_p_val: bool = True, return_distance: bool = True) -> Dict[Dict[str, str], Dict[str, Union[int, float]]]
```

Predict whether a batch of data has drifted from the reference data.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, list]` |  | Batch of instances. |
| `return_p_val` | `bool` | `True` | Whether to return the p-value of the permutation test. |
| `return_distance` | `bool` | `True` | Whether to return the MMD metric between the new batch and reference data. |

**Returns**
- Type: `Dict[Dict[str, str], Dict[str, Union[int, float]]]`

#### `score`

```python
score(x: Union[numpy.ndarray, list]) -> Tuple[float, float, float]
```

Compute the p-value resulting from a permutation test using the maximum mean discrepancy

as a distance measure between the reference data and the data to be tested.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, list]` |  | Batch of instances. |

**Returns**
- Type: `Tuple[float, float, float]`
