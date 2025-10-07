# `alibi_detect.od.pytorch.gmm`
## `GMMTorch`

_Inherits from:_ `TorchOutlierDetector`, `Module`, `FitMixinTorch`, `ABC`

### Constructor

```python
GMMTorch(self, n_components: int, device: Union[typing_extensions.Literal['cuda', 'gpu', 'cpu'], ForwardRef('torch.device'), NoneType] = None)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `n_components` | `int` |  | Number of components in gaussian mixture model. |
| `device` | `Union[Literal[cuda, gpu, cpu], torch.device, None]` | `None` | Device type used. The default tries to use the GPU and falls back on CPU if needed. Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of ``torch.device``. |

### Methods

#### `fit`

```python
fit(x_ref: torch.Tensor, optimizer: type[torch.optim.optimizer.Optimizer] = <class 'torch.optim.adam.Adam'>, learning_rate: float = 0.1, max_epochs: int = 10, batch_size: int = 32, tol: float = 0.001, n_iter_no_change: int = 25, verbose: int = 0) -> Dict
```

Fit the GMM model.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `torch.Tensor` |  | Training data. |
| `optimizer` | `type[torch.optim.optimizer.Optimizer]` | `<class 'torch.optim.adam.Adam'>` | Optimizer used to train the model. |
| `learning_rate` | `float` | `0.1` | Learning rate used to train the model. |
| `max_epochs` | `int` | `10` | Maximum number of training epochs. |
| `batch_size` | `int` | `32` | Batch size used to train the model. |
| `tol` | `float` | `0.001` | Convergence threshold. Training iterations will stop when the lower bound average gain is below this threshold. |
| `n_iter_no_change` | `int` | `25` | The number of iterations over which the loss must decrease by `tol` in order for optimization to continue. |
| `verbose` | `int` | `0` | Verbosity level during training. 0 is silent, 1 a progress bar. |

**Returns**
- Type: `Dict`

#### `format_fit_kwargs`

```python
format_fit_kwargs(fit_kwargs: Dict) -> Dict
```

Format kwargs for `fit` method.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `fit_kwargs` | `Dict` |  |  |
| `kwargs` |  |  | dictionary of Kwargs to format. See `fit` method for details. |

**Returns**
- Type: `Dict`

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
| `x` | `torch.Tensor` |  | `torch.Tensor` with leading batch dimension. |

**Returns**
- Type: `torch.Tensor`
