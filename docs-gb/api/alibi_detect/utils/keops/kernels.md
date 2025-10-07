# `alibi_detect.utils.keops.kernels`
## `DeepKernel`

_Inherits from:_ `Module`

### Constructor

```python
DeepKernel(self, proj: torch.nn.modules.module.Module, kernel_a: Union[torch.nn.modules.module.Module, typing_extensions.Literal['rbf']] = 'rbf', kernel_b: Union[torch.nn.modules.module.Module, typing_extensions.Literal['rbf'], NoneType] = 'rbf', eps: Union[float, typing_extensions.Literal['trainable']] = 'trainable') -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `proj` | `torch.nn.modules.module.Module` |  | The projection to be applied to the inputs before applying kernel_a |
| `kernel_a` | `Union[torch.nn.modules.module.Module, Literal[rbf]]` | `'rbf'` | The kernel to apply to the projected inputs. Defaults to a Gaussian RBF with trainable bandwidth. |
| `kernel_b` | `Union[torch.nn.modules.module.Module, Literal[rbf], None]` | `'rbf'` | The kernel to apply to the raw inputs. Defaults to a Gaussian RBF with trainable bandwidth. Set to None in order to use only the deep component (i.e. eps=0). |
| `eps` | `Union[float, Literal[trainable]]` | `'trainable'` | The proportion (in [0,1]) of weight to assign to the kernel applied to raw inputs. This can be either specified or set to 'trainable'. Only relavent if kernel_b is not None. |

### Properties

| Property | Type | Description |
| -------- | ---- | ----------- |
| `eps` | `torch.Tensor` |  |

### Methods

#### `forward`

```python
forward(x_proj: pykeops.torch.lazytensor.LazyTensor.LazyTensor, y_proj: pykeops.torch.lazytensor.LazyTensor.LazyTensor, x: Optional[pykeops.torch.lazytensor.LazyTensor.LazyTensor] = None, y: Optional[pykeops.torch.lazytensor.LazyTensor.LazyTensor] = None) -> pykeops.torch.lazytensor.LazyTensor.LazyTensor
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_proj` | `pykeops.torch.lazytensor.LazyTensor.LazyTensor` |  |  |
| `y_proj` | `pykeops.torch.lazytensor.LazyTensor.LazyTensor` |  |  |
| `x` | `Optional[pykeops.torch.lazytensor.LazyTensor.LazyTensor]` | `None` |  |
| `y` | `Optional[pykeops.torch.lazytensor.LazyTensor.LazyTensor]` | `None` |  |

**Returns**
- Type: `pykeops.torch.lazytensor.LazyTensor.LazyTensor`

#### `from_config`

```python
from_config(config)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `config` |  |  |  |

#### `get_config`

```python
get_config() -> dict
```

**Returns**
- Type: `dict`

## `GaussianRBF`

_Inherits from:_ `Module`

### Constructor

```python
GaussianRBF(self, sigma: Optional[torch.Tensor] = None, init_sigma_fn: Optional[Callable] = None, trainable: bool = False) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `sigma` | `Optional[torch.Tensor]` | `None` | Bandwidth used for the kernel. Needn't be specified if being inferred or trained. Can pass multiple values to eval kernel with and then average. |
| `init_sigma_fn` | `Optional[Callable]` | `None` | Function used to compute the bandwidth `sigma`. Used when `sigma` is to be inferred. The function's signature should match :py:func:`~alibi_detect.utils.keops.kernels.sigma_mean`, meaning that it should take in the lazy tensors `x`, `y` and `dist` and return a tensor `sigma`. |
| `trainable` | `bool` | `False` | Whether or not to track gradients w.r.t. `sigma` to allow it to be trained. |

### Properties

| Property | Type | Description |
| -------- | ---- | ----------- |
| `sigma` | `torch.Tensor` |  |

### Methods

#### `forward`

```python
forward(x: pykeops.torch.lazytensor.LazyTensor.LazyTensor, y: pykeops.torch.lazytensor.LazyTensor.LazyTensor, infer_sigma: bool = False) -> pykeops.torch.lazytensor.LazyTensor.LazyTensor
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `pykeops.torch.lazytensor.LazyTensor.LazyTensor` |  |  |
| `y` | `pykeops.torch.lazytensor.LazyTensor.LazyTensor` |  |  |
| `infer_sigma` | `bool` | `False` |  |

**Returns**
- Type: `pykeops.torch.lazytensor.LazyTensor.LazyTensor`

#### `from_config`

```python
from_config(config)
```

Instantiates a kernel from a config dictionary.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `config` |  |  | A kernel config dictionary. |

#### `get_config`

```python
get_config() -> dict
```

Returns a serializable config dict (excluding the input_sigma_fn, which is serialized in alibi_detect.saving).

**Returns**
- Type: `dict`

## Functions
### `sigma_mean`

```python
sigma_mean(x: pykeops.torch.lazytensor.LazyTensor.LazyTensor, y: pykeops.torch.lazytensor.LazyTensor.LazyTensor, dist: pykeops.torch.lazytensor.LazyTensor.LazyTensor, n_min: int = 100) -> torch.Tensor
```

Set bandwidth to the mean distance between instances x and y.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `pykeops.torch.lazytensor.LazyTensor.LazyTensor` |  | LazyTensor of instances with dimension [Nx, 1, features] or [batch_size, Nx, 1, features]. The singleton dimension is necessary for broadcasting. |
| `y` | `pykeops.torch.lazytensor.LazyTensor.LazyTensor` |  | LazyTensor of instances with dimension [1, Ny, features] or [batch_size, 1, Ny, features]. The singleton dimension is necessary for broadcasting. |
| `dist` | `pykeops.torch.lazytensor.LazyTensor.LazyTensor` |  | LazyTensor with dimensions [Nx, Ny] or [batch_size, Nx, Ny] containing the pairwise distances between `x` and `y`. |
| `n_min` | `int` | `100` | In order to check whether x equals y after squeezing the singleton dimensions, we check if the diagonal of the distance matrix (which is a lazy tensor from which the diagonal cannot be directly extracted) consists of all zeros. We do this by computing the k-min distances and k-argmin indices over the columns of the distance matrix. We then check if the distances on the diagonal of the distance matrix are all zero or not. If they are all zero, then we do not use these distances (zeros) when computing the mean pairwise distance as bandwidth. If Nx becomes very large, it is advised to set `n_min` to a low enough value to avoid OOM issues. By default we set it to 100 instances. |

**Returns**
- Type: `torch.Tensor`
