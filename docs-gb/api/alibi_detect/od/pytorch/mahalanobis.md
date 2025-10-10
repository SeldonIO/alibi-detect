# `alibi_detect.od.pytorch.mahalanobis`
## `MahalanobisTorch`

_Inherits from:_ `TorchOutlierDetector`, `Module`, `FitMixinTorch`, `ABC`

### Constructor

```python
MahalanobisTorch(self, min_eigenvalue: float = 1e-06, device: Union[typing_extensions.Literal['cuda', 'gpu', 'cpu'], ForwardRef('torch.device'), NoneType] = None)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `min_eigenvalue` | `float` | `1e-06` | Eigenvectors with eigenvalues below this value will be discarded. |
| `device` | `Union[Literal[cuda, gpu, cpu], torch.device, None]` | `None` | Device type used. The default tries to use the GPU and falls back on CPU if needed. Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of ``torch.device``. |

### Methods

#### `fit`

```python
fit(x_ref: torch.Tensor)
```

Fits the detector

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `torch.Tensor` |  | The Dataset tensor. |

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
