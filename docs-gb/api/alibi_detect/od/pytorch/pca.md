# `alibi_detect.od.pytorch.pca`
## `KernelPCATorch`

_Inherits from:_ `PCATorch`, `TorchOutlierDetector`, `Module`, `FitMixinTorch`, `ABC`

### Constructor

```python
KernelPCATorch(self, n_components: int, kernel: Optional[Callable], device: Union[typing_extensions.Literal['cuda', 'gpu', 'cpu'], ForwardRef('torch.device'), NoneType] = None)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `n_components` | `int` |  | The number of dimensions in the principal subspace. |
| `kernel` | `Optional[Callable]` |  | Kernel function to use for outlier detection. |
| `device` | `Union[Literal[cuda, gpu, cpu], torch.device, None]` | `None` | Device type used. The default tries to use the GPU and falls back on CPU if needed. Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of ``torch.device``. |

### Methods

#### `compute_kernel_mat`

```python
compute_kernel_mat(x: torch.Tensor) -> torch.Tensor
```

Computes the centered kernel matrix.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `torch.Tensor` |  | The reference data. |

**Returns**
- Type: `torch.Tensor`

## `LinearPCATorch`

_Inherits from:_ `PCATorch`, `TorchOutlierDetector`, `Module`, `FitMixinTorch`, `ABC`

### Constructor

```python
LinearPCATorch(self, n_components: int, device: Union[typing_extensions.Literal['cuda', 'gpu', 'cpu'], ForwardRef('torch.device'), NoneType] = None)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `n_components` | `int` |  | The number of dimensions in the principal subspace. |
| `device` | `Union[Literal[cuda, gpu, cpu], torch.device, None]` | `None` | Device type used. The default tries to use the GPU and falls back on CPU if needed. Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of ``torch.device``. |

## `PCATorch`

_Inherits from:_ `TorchOutlierDetector`, `Module`, `FitMixinTorch`, `ABC`

### Constructor

```python
PCATorch(self, n_components: int, device: Union[typing_extensions.Literal['cuda', 'gpu', 'cpu'], ForwardRef('torch.device'), NoneType] = None)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `n_components` | `int` |  | The number of dimensions in the principal subspace. For linear PCA should have ``1 <= n_components < dim(data)``. For kernel pca should have ``1 <= n_components < len(data)``. |
| `device` | `Union[Literal[cuda, gpu, cpu], torch.device, None]` | `None` | Device type used. The default tries to use the GPU and falls back on CPU if needed. Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of ``torch.device``. |

### Methods

#### `fit`

```python
fit(x_ref: torch.Tensor) -> None
```

Fits the PCA detector.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `torch.Tensor` |  | The Dataset tensor. |

**Returns**
- Type: `None`

#### `forward`

```python
forward(x: torch.Tensor) -> torch.Tensor
```

Detect if `x` is an outlier.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `torch.Tensor` |  | `torch.Tensor` with leading batch dimension. |

**Returns**
- Type: `torch.Tensor`

#### `score`

```python
score(x: torch.Tensor) -> torch.Tensor
```

Computes the score of `x`

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `torch.Tensor` |  | The tensor of instances. First dimension corresponds to batch. |

**Returns**
- Type: `torch.Tensor`
