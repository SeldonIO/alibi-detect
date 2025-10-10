# `alibi_detect.od.pytorch.knn`
## `KNNTorch`

_Inherits from:_ `TorchOutlierDetector`, `Module`, `FitMixinTorch`, `ABC`

### Constructor

```python
KNNTorch(self, k: Union[numpy.ndarray, List, Tuple, int], kernel: Optional[torch.nn.modules.module.Module] = None, ensembler: Optional[alibi_detect.od.pytorch.ensemble.Ensembler] = None, device: Union[typing_extensions.Literal['cuda', 'gpu', 'cpu'], ForwardRef('torch.device'), NoneType] = None)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `k` | `Union[numpy.ndarray, List[Any], Tuple, int]` |  | Number of nearest neighbors to compute distance to. `k` can be a single value or an array of integers. If `k` is a single value the outlier score is the distance/kernel similarity to the `k`-th nearest neighbor. If `k` is a list then it returns the distance/kernel similarity to each of the specified `k` neighbors. |
| `kernel` | `Optional[torch.nn.modules.module.Module]` | `None` | If a kernel is specified then instead of using `torch.cdist` the kernel defines the `k` nearest neighbor distance. |
| `ensembler` | `Optional[alibi_detect.od.pytorch.ensemble.Ensembler]` | `None` | If `k` is an array of integers then the ensembler must not be ``None``. Should be an instance of :py:obj:`alibi_detect.od.pytorch.ensemble.ensembler`. Responsible for combining multiple scores into a single score. |
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
