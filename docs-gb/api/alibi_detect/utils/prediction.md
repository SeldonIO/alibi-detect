# `alibi_detect.utils.prediction`
## Functions
### `tokenize_transformer`

```python
tokenize_transformer(x: Union[list, numpy.ndarray], tokenizer: Callable, max_len: int, backend: str) -> dict
```

Batch tokenizer for transformer models.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[list, numpy.ndarray]` |  | Batch of instances. |
| `tokenizer` | `Callable` |  | Tokenizer for model. |
| `max_len` | `int` |  | Max token length. |
| `backend` | `str` |  | PyTorch ('pt') or TensorFlow ('tf') backend. |

**Returns**
- Type: `dict`
