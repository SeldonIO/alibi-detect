# `alibi_detect.cd.pytorch.context_aware`
## Constants
### `logger`
```python
logger: logging.Logger = <Logger alibi_detect.cd.pytorch.context_aware (WARNING)>
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

## `ContextMMDDriftTorch`

_Inherits from:_ `BaseContextMMDDrift`, `BaseDetector`, `ABC`

### Constructor

```python
ContextMMDDriftTorch(self, x_ref: Union[numpy.ndarray, list], c_ref: numpy.ndarray, p_val: float = 0.05, x_ref_preprocessed: bool = False, preprocess_at_init: bool = True, update_ref: Optional[Dict[str, int]] = None, preprocess_fn: Optional[Callable] = None, x_kernel: Callable = <class 'alibi_detect.utils.pytorch.kernels.GaussianRBF'>, c_kernel: Callable = <class 'alibi_detect.utils.pytorch.kernels.GaussianRBF'>, n_permutations: int = 1000, prop_c_held: float = 0.25, n_folds: int = 5, batch_size: Optional[int] = 256, device: Union[typing_extensions.Literal['cuda', 'gpu', 'cpu'], ForwardRef('torch.device'), NoneType] = None, input_shape: Optional[tuple] = None, data_type: Optional[str] = None, verbose: bool = False) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `Union[numpy.ndarray, list]` |  | Data used as reference distribution. |
| `c_ref` | `numpy.ndarray` |  | Context for the reference distribution. |
| `p_val` | `float` | `0.05` | p-value used for the significance of the permutation test. |
| `x_ref_preprocessed` | `bool` | `False` | Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference data will also be preprocessed. |
| `preprocess_at_init` | `bool` | `True` | Whether to preprocess the reference data when the detector is instantiated. Otherwise, the reference data will be preprocessed at prediction time. Only applies if `x_ref_preprocessed=False`. |
| `update_ref` | `Optional[Dict[str, int]]` | `None` | Reference data can optionally be updated to the last N instances seen by the detector. The parameter should be passed as a dictionary *{'last': N}*. |
| `preprocess_fn` | `Optional[Callable]` | `None` | Function to preprocess the data before computing the data drift metrics. |
| `x_kernel` | `Callable` | `<class 'alibi_detect.utils.pytorch.kernels.GaussianRBF'>` | Kernel defined on the input data, defaults to Gaussian RBF kernel. |
| `c_kernel` | `Callable` | `<class 'alibi_detect.utils.pytorch.kernels.GaussianRBF'>` | Kernel defined on the context data, defaults to Gaussian RBF kernel. |
| `n_permutations` | `int` | `1000` | Number of permutations used in the permutation test. |
| `prop_c_held` | `float` | `0.25` | Proportion of contexts held out to condition on. |
| `n_folds` | `int` | `5` | Number of cross-validation folds used when tuning the regularisation parameters. |
| `batch_size` | `Optional[int]` | `256` | If not None, then compute batches of MMDs at a time (rather than all at once). |
| `device` | `Union[Literal[cuda, gpu, cpu], torch.device, None]` | `None` | Device type used. The default tries to use the GPU and falls back on CPU if needed. Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of ``torch.device``. Only relevant for 'pytorch' backend. |
| `input_shape` | `Optional[tuple]` | `None` | Shape of input data. |
| `data_type` | `Optional[str]` | `None` | Optionally specify the data type (tabular, image or time-series). Added to metadata. |
| `verbose` | `bool` | `False` | Whether or not to print progress during configuration. |

### Methods

#### `score`

```python
score(x: Union[numpy.ndarray, list], c: numpy.ndarray) -> Tuple[float, float, float, Tuple]
```

Compute the MMD based conditional test statistic, and perform a conditional permutation test to obtain a

p-value representing the test statistic's extremity under the null hypothesis.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, list]` |  | Batch of instances. |
| `c` | `numpy.ndarray` |  | Context associated with batch of instances. |

**Returns**
- Type: `Tuple[float, float, float, Tuple]`
