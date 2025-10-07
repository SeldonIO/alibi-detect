# `alibi_detect.utils.tensorflow.distance`
## Constants
### `logger`
```python
logger: logging.Logger = <Logger alibi_detect.utils.tensorflow.distance (WARNING)>
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

## Functions
### `batch_compute_kernel_matrix`

```python
batch_compute_kernel_matrix(x: Union[list, numpy.ndarray, tensorflow.python.framework.tensor.Tensor], y: Union[list, numpy.ndarray, tensorflow.python.framework.tensor.Tensor], kernel: Union[Callable, keras.src.models.model.Model], batch_size: int = 10000000000, preprocess_fn: Optional[Callable] = None) -> tensorflow.python.framework.tensor.Tensor
```

Compute the kernel matrix between x and y by filling in blocks of size

batch_size x batch_size at a time.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[list, numpy.ndarray, tensorflow.python.framework.tensor.Tensor]` |  | Reference set. |
| `y` | `Union[list, numpy.ndarray, tensorflow.python.framework.tensor.Tensor]` |  | Test set. |
| `kernel` | `Union[Callable, keras.src.models.model.Model]` |  | tf.keras model |
| `batch_size` | `int` | `10000000000` | Batch size used during prediction. |
| `preprocess_fn` | `Optional[Callable]` | `None` | Optional preprocessing function for each batch. |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`

### `mmd2`

```python
mmd2(x: tensorflow.python.framework.tensor.Tensor, y: tensorflow.python.framework.tensor.Tensor, kernel: Callable) -> float
```

Compute MMD^2 between 2 samples.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `tensorflow.python.framework.tensor.Tensor` |  | Batch of instances of shape [Nx, features]. |
| `y` | `tensorflow.python.framework.tensor.Tensor` |  | Batch of instances of shape [Ny, features]. |
| `kernel` | `Callable` |  | Kernel function. |

**Returns**
- Type: `float`

### `mmd2_from_kernel_matrix`

```python
mmd2_from_kernel_matrix(kernel_mat: tensorflow.python.framework.tensor.Tensor, m: int, permute: bool = False, zero_diag: bool = True) -> tensorflow.python.framework.tensor.Tensor
```

Compute maximum mean discrepancy (MMD^2) between 2 samples x and y from the

full kernel matrix between the samples.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `kernel_mat` | `tensorflow.python.framework.tensor.Tensor` |  | Kernel matrix between samples x and y. |
| `m` | `int` |  | Number of instances in y. |
| `permute` | `bool` | `False` | Whether to permute the row indices. Used for permutation tests. |
| `zero_diag` | `bool` | `True` | Whether to zero out the diagonal of the kernel matrix. |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`

### `permed_lsdds`

```python
permed_lsdds(k_all_c: tensorflow.python.framework.tensor.Tensor, x_perms: List[tensorflow.python.framework.tensor.Tensor], y_perms: List[tensorflow.python.framework.tensor.Tensor], H: tensorflow.python.framework.tensor.Tensor, H_lam_inv: Optional[tensorflow.python.framework.tensor.Tensor] = None, lam_rd_max: float = 0.2, return_unpermed: bool = False) -> Union[Tuple[tensorflow.python.framework.tensor.Tensor, tensorflow.python.framework.tensor.Tensor], Tuple[tensorflow.python.framework.tensor.Tensor, tensorflow.python.framework.tensor.Tensor, tensorflow.python.framework.tensor.Tensor]]
```

Compute LSDD estimates from kernel matrix across various ref and test window samples

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `k_all_c` | `tensorflow.python.framework.tensor.Tensor` |  | Kernel matrix of similarities between all samples and the kernel centers. |
| `x_perms` | `List[tensorflow.python.framework.tensor.Tensor]` |  | List of B reference window index vectors |
| `y_perms` | `List[tensorflow.python.framework.tensor.Tensor]` |  | List of B test window index vectors |
| `H` | `tensorflow.python.framework.tensor.Tensor` |  | Special (scaled) kernel matrix of similarities between kernel centers |
| `H_lam_inv` | `Optional[tensorflow.python.framework.tensor.Tensor]` | `None` | Function of H corresponding to a particular regulariation parameter lambda. See Eqn 11 of Bu et al. (2017) |
| `lam_rd_max` | `float` | `0.2` | The maximum relative difference between two estimates of LSDD that the regularization parameter lambda is allowed to cause. Defaults to 0.2. Only relavent if H_lam_inv is not supplied. |
| `return_unpermed` | `bool` | `False` | Whether or not to return value corresponding to unpermed order defined by k_all_c |

**Returns**
- Type: `Union[Tuple[tensorflow.python.framework.tensor.Tensor, tensorflow.python.framework.tensor.Tensor], Tuple[tensorflow.python.framework.tensor.Tensor, tensorflow.python.framework.tensor.Tensor, tensorflow.python.framework.tensor.Tensor]]`

### `relative_euclidean_distance`

```python
relative_euclidean_distance(x: tensorflow.python.framework.tensor.Tensor, y: tensorflow.python.framework.tensor.Tensor, eps: float = 1e-12, axis: int = -1) -> tensorflow.python.framework.tensor.Tensor
```

Relative Euclidean distance.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `tensorflow.python.framework.tensor.Tensor` |  | Tensor used in distance computation. |
| `y` | `tensorflow.python.framework.tensor.Tensor` |  | Tensor used in distance computation. |
| `eps` | `float` | `1e-12` | Epsilon added to denominator for numerical stability. |
| `axis` | `int` | `-1` | Axis used to compute distance. |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`

### `squared_pairwise_distance`

```python
squared_pairwise_distance(x: tensorflow.python.framework.tensor.Tensor, y: tensorflow.python.framework.tensor.Tensor, a_min: float = 1e-30, a_max: float = 1e+30) -> tensorflow.python.framework.tensor.Tensor
```

TensorFlow pairwise squared Euclidean distance between samples x and y.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `tensorflow.python.framework.tensor.Tensor` |  | Batch of instances of shape [Nx, features]. |
| `y` | `tensorflow.python.framework.tensor.Tensor` |  | Batch of instances of shape [Ny, features]. |
| `a_min` | `float` | `1e-30` | Lower bound to clip distance values. |
| `a_max` | `float` | `1e+30` | Upper bound to clip distance values. |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`
