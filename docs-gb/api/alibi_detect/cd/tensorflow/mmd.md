# `alibi_detect.cd.tensorflow.mmd`
## Constants
### `logger`
```python
logger: logging.Logger = <Logger alibi_detect.cd.tensorflow.mmd (WARNING)>
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

## `MMDDriftTF`

_Inherits from:_ `BaseMMDDrift`, `BaseDetector`, `ABC`

### Constructor

```python
MMDDriftTF(self, x_ref: Union[numpy.ndarray, list], p_val: float = 0.05, x_ref_preprocessed: bool = False, preprocess_at_init: bool = True, update_x_ref: Optional[Dict[str, int]] = None, preprocess_fn: Optional[Callable] = None, kernel: Callable = <class 'alibi_detect.utils.tensorflow.kernels.GaussianRBF'>, sigma: Optional[numpy.ndarray] = None, configure_kernel_from_x_ref: bool = True, n_permutations: int = 100, input_shape: Optional[tuple] = None, data_type: Optional[str] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `Union[numpy.ndarray, list]` |  | Data used as reference distribution. |
| `p_val` | `float` | `0.05` | p-value used for the significance of the permutation test. |
| `x_ref_preprocessed` | `bool` | `False` | Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference data will also be preprocessed. |
| `preprocess_at_init` | `bool` | `True` | Whether to preprocess the reference data when the detector is instantiated. Otherwise, the reference data will be preprocessed at prediction time. Only applies if `x_ref_preprocessed=False`. |
| `update_x_ref` | `Optional[Dict[str, int]]` | `None` | Reference data can optionally be updated to the last n instances seen by the detector or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while for reservoir sampling {'reservoir_sampling': n} is passed. |
| `preprocess_fn` | `Optional[Callable]` | `None` | Function to preprocess the data before computing the data drift metrics. |
| `kernel` | `Callable` | `<class 'alibi_detect.utils.tensorflow.kernels.GaussianRBF'>` | Kernel used for the MMD computation, defaults to Gaussian RBF kernel. |
| `sigma` | `Optional[numpy.ndarray]` | `None` | Optionally set the GaussianRBF kernel bandwidth. Can also pass multiple bandwidth values as an array. The kernel evaluation is then averaged over those bandwidths. |
| `configure_kernel_from_x_ref` | `bool` | `True` | Whether to already configure the kernel bandwidth from the reference data. |
| `n_permutations` | `int` | `100` | Number of permutations used in the permutation test. |
| `input_shape` | `Optional[tuple]` | `None` | Shape of input data. |
| `data_type` | `Optional[str]` | `None` | Optionally specify the data type (tabular, image or time-series). Added to metadata. |

### Methods

#### `kernel_matrix`

```python
kernel_matrix(x: Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor], y: Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor]) -> tensorflow.python.framework.tensor.Tensor
```

Compute and return full kernel matrix between arrays x and y.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor]` |  |  |
| `y` | `Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor]` |  |  |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`

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
