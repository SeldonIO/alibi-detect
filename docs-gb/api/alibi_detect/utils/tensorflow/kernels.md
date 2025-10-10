# `alibi_detect.utils.tensorflow.kernels`
## `DeepKernel`

_Inherits from:_ `Model`, `TensorFlowTrainer`, `Trainer`, `Layer`, `TFLayer`, `KerasAutoTrackable`, `AutoTrackable`, `Trackable`, `Operation`, `KerasSaveable`

Computes similarities as k(x,y) = (1-eps)*k_a(proj(x), proj(y)) + eps*k_b(x,y).

A forward pass takes a batch of instances x [Nx, features] and y [Ny, features] and returns
the kernel matrix [Nx, Ny].

### Constructor

```python
DeepKernel(self, proj: keras.src.models.model.Model, kernel_a: Union[keras.src.models.model.Model, str] = 'rbf', kernel_b: Union[keras.src.models.model.Model, str, NoneType] = 'rbf', eps: Union[float, str] = 'trainable') -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `proj` | `keras.src.models.model.Model` |  |  |
| `kernel_a` | `Union[keras.src.models.model.Model, str]` | `'rbf'` |  |
| `kernel_b` | `Union[keras.src.models.model.Model, str, None]` | `'rbf'` |  |
| `eps` | `Union[float, str]` | `'trainable'` |  |

### Properties

| Property | Type | Description |
| -------- | ---- | ----------- |
| `eps` | `tensorflow.python.framework.tensor.Tensor` |  |

### Methods

#### `call`

```python
call(x: tensorflow.python.framework.tensor.Tensor, y: tensorflow.python.framework.tensor.Tensor) -> tensorflow.python.framework.tensor.Tensor
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `tensorflow.python.framework.tensor.Tensor` |  |  |
| `y` | `tensorflow.python.framework.tensor.Tensor` |  |  |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`

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

_Inherits from:_ `Model`, `TensorFlowTrainer`, `Trainer`, `Layer`, `TFLayer`, `KerasAutoTrackable`, `AutoTrackable`, `Trackable`, `Operation`, `KerasSaveable`

### Constructor

```python
GaussianRBF(self, sigma: Optional[tensorflow.python.framework.tensor.Tensor] = None, init_sigma_fn: Optional[Callable] = None, trainable: bool = False) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `sigma` | `Optional[tensorflow.python.framework.tensor.Tensor]` | `None` | Bandwidth used for the kernel. Needn't be specified if being inferred or trained. Can pass multiple values to eval kernel with and then average. |
| `init_sigma_fn` | `Optional[Callable]` | `None` | Function used to compute the bandwidth `sigma`. Used when `sigma` is to be inferred. The function's signature should match :py:func:`~alibi_detect.utils.tensorflow.kernels.sigma_median`, meaning that it should take in the tensors `x`, `y` and `dist` and return `sigma`. If `None`, it is set to :func:`~alibi_detect.utils.tensorflow.kernels.sigma_median`. |
| `trainable` | `bool` | `False` | Whether or not to track gradients w.r.t. sigma to allow it to be trained. |

### Properties

| Property | Type | Description |
| -------- | ---- | ----------- |
| `sigma` | `tensorflow.python.framework.tensor.Tensor` |  |

### Methods

#### `call`

```python
call(x: tensorflow.python.framework.tensor.Tensor, y: tensorflow.python.framework.tensor.Tensor, infer_sigma: bool = False) -> tensorflow.python.framework.tensor.Tensor
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `tensorflow.python.framework.tensor.Tensor` |  |  |
| `y` | `tensorflow.python.framework.tensor.Tensor` |  |  |
| `infer_sigma` | `bool` | `False` |  |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`

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
### `sigma_median`

```python
sigma_median(x: tensorflow.python.framework.tensor.Tensor, y: tensorflow.python.framework.tensor.Tensor, dist: tensorflow.python.framework.tensor.Tensor) -> tensorflow.python.framework.tensor.Tensor
```

Bandwidth estimation using the median heuristic :cite:t:`Gretton2012`.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `tensorflow.python.framework.tensor.Tensor` |  | Tensor of instances with dimension [Nx, features]. |
| `y` | `tensorflow.python.framework.tensor.Tensor` |  | Tensor of instances with dimension [Ny, features]. |
| `dist` | `tensorflow.python.framework.tensor.Tensor` |  | Tensor with dimensions [Nx, Ny], containing the pairwise distances between `x` and `y`. |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`
