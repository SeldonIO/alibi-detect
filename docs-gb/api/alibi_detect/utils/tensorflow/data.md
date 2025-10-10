# `alibi_detect.utils.tensorflow.data`
## `TFDataset`

_Inherits from:_ `PyDataset`

### Constructor

```python
TFDataset(self, *indexables: Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor, list], batch_size: int = 10000000000, shuffle: bool = True) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `batch_size` | `int` | `10000000000` |  |
| `shuffle` | `bool` | `True` |  |

### Methods

#### `on_epoch_end`

```python
on_epoch_end() -> None
```

**Returns**
- Type: `None`
