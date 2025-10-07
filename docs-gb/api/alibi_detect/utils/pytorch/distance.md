# `alibi_detect.utils.pytorch.distance`
## Constants
### `logger`
```python
logger: logging.Logger = <Logger alibi_detect.utils.pytorch.distance (WARNING)>
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
batch_compute_kernel_matrix(x: Union[list, numpy.ndarray, torch.Tensor], y: Union[list, numpy.ndarray, torch.Tensor], kernel: Union[torch.nn.modules.module.Module, torch.nn.modules.container.Sequential], device: Optional[torch.device] = None, batch_size: int = 10000000000, preprocess_fn: Optional[Callable[[...], torch.Tensor]] = None) -> torch.Tensor
```

Compute the kernel matrix between x and y by filling in blocks of size

batch_size x batch_size at a time.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[list, numpy.ndarray, torch.Tensor]` |  | Reference set. |
| `y` | `Union[list, numpy.ndarray, torch.Tensor]` |  | Test set. |
| `kernel` | `Union[torch.nn.modules.module.Module, torch.nn.modules.container.Sequential]` |  | PyTorch module. |
| `device` | `Optional[torch.device]` | `None` | Device type used. The default None tries to use the GPU and falls back on CPU if needed. Can be specified by passing either torch.device('cuda') or torch.device('cpu'). |
| `batch_size` | `int` | `10000000000` | Batch size used during prediction. |
| `preprocess_fn` | `Optional[Callable[[...], torch.Tensor]]` | `None` | Optional preprocessing function for each batch. |

**Returns**
- Type: `torch.Tensor`

### `mmd2`

```python
mmd2(x: torch.Tensor, y: torch.Tensor, kernel: Callable) -> float
```

Compute MMD^2 between 2 samples.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `torch.Tensor` |  | Batch of instances of shape [Nx, features]. |
| `y` | `torch.Tensor` |  | Batch of instances of shape [Ny, features]. |
| `kernel` | `Callable` |  | Kernel function. |

**Returns**
- Type: `float`

### `mmd2_from_kernel_matrix`

```python
mmd2_from_kernel_matrix(kernel_mat: torch.Tensor, m: int, permute: bool = False, zero_diag: bool = True) -> torch.Tensor
```

Compute maximum mean discrepancy (MMD^2) between 2 samples x and y from the

full kernel matrix between the samples.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `kernel_mat` | `torch.Tensor` |  | Kernel matrix between samples x and y. |
| `m` | `int` |  | Number of instances in y. |
| `permute` | `bool` | `False` | Whether to permute the row indices. Used for permutation tests. |
| `zero_diag` | `bool` | `True` | Whether to zero out the diagonal of the kernel matrix. |

**Returns**
- Type: `torch.Tensor`

### `permed_lsdds`

```python
permed_lsdds(k_all_c: torch.Tensor, x_perms: List[torch.Tensor], y_perms: List[torch.Tensor], H: torch.Tensor, H_lam_inv: Optional[torch.Tensor] = None, lam_rd_max: float = 0.2, return_unpermed: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
```

Compute LSDD estimates from kernel matrix across various ref and test window samples

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `k_all_c` | `torch.Tensor` |  | Kernel matrix of similarities between all samples and the kernel centers. |
| `x_perms` | `List[torch.Tensor]` |  | List of B reference window index vectors |
| `y_perms` | `List[torch.Tensor]` |  | List of B test window index vectors |
| `H` | `torch.Tensor` |  | Special (scaled) kernel matrix of similarities between kernel centers |
| `H_lam_inv` | `Optional[torch.Tensor]` | `None` | Function of H corresponding to a particular regulariation parameter lambda. See Eqn 11 of Bu et al. (2017) |
| `lam_rd_max` | `float` | `0.2` | The maximum relative difference between two estimates of LSDD that the regularization parameter lambda is allowed to cause. Defaults to 0.2. Only relavent if H_lam_inv is not supplied. |
| `return_unpermed` | `bool` | `False` | Whether or not to return value corresponding to unpermed order defined by k_all_c |

**Returns**
- Type: `Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]`
