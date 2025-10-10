# `alibi_detect.cd.pytorch.preprocess`
## `HiddenOutput`

_Inherits from:_ `Module`

### Constructor

```python
HiddenOutput(self, model: Union[torch.nn.modules.module.Module, torch.nn.modules.container.Sequential], layer: int = -1, flatten: bool = False) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `model` | `Union[torch.nn.modules.module.Module, torch.nn.modules.container.Sequential]` |  |  |
| `layer` | `int` | `-1` |  |
| `flatten` | `bool` | `False` |  |

### Methods

#### `forward`

```python
forward(x: torch.Tensor) -> torch.Tensor
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `torch.Tensor` |  |  |

**Returns**
- Type: `torch.Tensor`

## `UAE`

_Inherits from:_ `Module`

### Constructor

```python
UAE(self, encoder_net: Optional[torch.nn.modules.module.Module] = None, input_layer: Optional[torch.nn.modules.module.Module] = None, shape: Optional[tuple] = None, enc_dim: Optional[int] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `encoder_net` | `Optional[torch.nn.modules.module.Module]` | `None` |  |
| `input_layer` | `Optional[torch.nn.modules.module.Module]` | `None` |  |
| `shape` | `Optional[tuple]` | `None` |  |
| `enc_dim` | `Optional[int]` | `None` |  |

### Methods

#### `forward`

```python
forward(x: Union[numpy.ndarray, torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, torch.Tensor, Dict[str, torch.Tensor]]` |  |  |

**Returns**
- Type: `torch.Tensor`

## Functions
### `preprocess_drift`

```python
preprocess_drift(x: Union[numpy.ndarray, list], model: Union[torch.nn.modules.module.Module, torch.nn.modules.container.Sequential], device: Union[Literal[cuda, gpu, cpu], torch.device, None] = None, preprocess_batch_fn: Optional[Callable] = None, tokenizer: Optional[Callable] = None, max_len: Optional[int] = None, batch_size: int = 10000000000, dtype: Union[type[numpy.generic], torch.dtype] = <class 'numpy.float32'>) -> Union[numpy.ndarray, torch.Tensor, tuple]
```

Prediction function used for preprocessing step of drift detector.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, list]` |  | Batch of instances. |
| `model` | `Union[torch.nn.modules.module.Module, torch.nn.modules.container.Sequential]` |  | Model used for preprocessing. |
| `device` | `Union[Literal[cuda, gpu, cpu], torch.device, None]` | `None` | Device type used. The default tries to use the GPU and falls back on CPU if needed. Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of ``torch.device``. |
| `preprocess_batch_fn` | `Optional[Callable]` | `None` | Optional batch preprocessing function. For example to convert a list of objects to a batch which can be processed by the PyTorch model. |
| `tokenizer` | `Optional[Callable]` | `None` | Optional tokenizer for text drift. |
| `max_len` | `Optional[int]` | `None` | Optional max token length for text drift. |
| `batch_size` | `int` | `10000000000` | Batch size used during prediction. |
| `dtype` | `Union[type[numpy.generic], torch.dtype]` | `<class 'numpy.float32'>` | Model output type, e.g. np.float32 or torch.float32. |

**Returns**
- Type: `Union[numpy.ndarray, torch.Tensor, tuple]`
