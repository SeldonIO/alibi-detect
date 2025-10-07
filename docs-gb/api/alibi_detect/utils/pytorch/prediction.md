# `alibi_detect.utils.pytorch.prediction`
## Functions
### `predict_batch`

```python
predict_batch(x: Union[list, numpy.ndarray, torch.Tensor], model: Union[Callable, torch.nn.modules.module.Module, torch.nn.modules.container.Sequential], device: Union[Literal[cuda, gpu, cpu], torch.device, None] = None, batch_size: int = 10000000000, preprocess_fn: Optional[Callable] = None, dtype: Union[type[numpy.generic], torch.dtype] = <class 'numpy.float32'>) -> Union[numpy.ndarray, torch.Tensor, tuple]
```

Make batch predictions on a model.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[list, numpy.ndarray, torch.Tensor]` |  | Batch of instances. |
| `model` | `Union[Callable, torch.nn.modules.module.Module, torch.nn.modules.container.Sequential]` |  | PyTorch model. |
| `device` | `Union[Literal[cuda, gpu, cpu], torch.device, None]` | `None` | Device type used. The default tries to use the GPU and falls back on CPU if needed. Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of ``torch.device``. |
| `batch_size` | `int` | `10000000000` | Batch size used during prediction. |
| `preprocess_fn` | `Optional[Callable]` | `None` | Optional preprocessing function for each batch. |
| `dtype` | `Union[type[numpy.generic], torch.dtype]` | `<class 'numpy.float32'>` | Model output type, e.g. np.float32 or torch.float32. |

**Returns**
- Type: `Union[numpy.ndarray, torch.Tensor, tuple]`

### `predict_batch_transformer`

```python
predict_batch_transformer(x: Union[list, numpy.ndarray], model: Union[torch.nn.modules.module.Module, torch.nn.modules.container.Sequential], tokenizer: Callable, max_len: int, device: Union[Literal[cuda, gpu, cpu], torch.device, None] = None, batch_size: int = 10000000000, dtype: Union[type[numpy.generic], torch.dtype] = <class 'numpy.float32'>) -> Union[numpy.ndarray, torch.Tensor, tuple]
```

Make batch predictions using a transformers tokenizer and model.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[list, numpy.ndarray]` |  | Batch of instances. |
| `model` | `Union[torch.nn.modules.module.Module, torch.nn.modules.container.Sequential]` |  | PyTorch model. |
| `tokenizer` | `Callable` |  | Tokenizer for model. |
| `max_len` | `int` |  | Max sequence length for tokens. |
| `device` | `Union[Literal[cuda, gpu, cpu], torch.device, None]` | `None` | Device type used. The default tries to use the GPU and falls back on CPU if needed. Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of ``torch.device``. |
| `batch_size` | `int` | `10000000000` | Batch size used during prediction. |
| `dtype` | `Union[type[numpy.generic], torch.dtype]` | `<class 'numpy.float32'>` | Model output type, e.g. np.float32 or torch.float32. |

**Returns**
- Type: `Union[numpy.ndarray, torch.Tensor, tuple]`
