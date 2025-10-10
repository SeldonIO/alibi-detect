# `alibi_detect.cd.tensorflow.preprocess`
## `HiddenOutput`

_Inherits from:_ `Model`, `TensorFlowTrainer`, `Trainer`, `Layer`, `TFLayer`, `KerasAutoTrackable`, `AutoTrackable`, `Trackable`, `Operation`, `KerasSaveable`

### Constructor

```python
HiddenOutput(self, model: keras.src.models.model.Model, layer: int = -1, input_shape: tuple = None, flatten: bool = False) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `model` | `keras.src.models.model.Model` |  |  |
| `layer` | `int` | `-1` |  |
| `input_shape` | `Optional[tuple]` | `None` |  |
| `flatten` | `bool` | `False` |  |

### Methods

#### `call`

```python
call(x: Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor]) -> tensorflow.python.framework.tensor.Tensor
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor]` |  |  |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`

## `UAE`

_Inherits from:_ `Model`, `TensorFlowTrainer`, `Trainer`, `Layer`, `TFLayer`, `KerasAutoTrackable`, `AutoTrackable`, `Trackable`, `Operation`, `KerasSaveable`

### Constructor

```python
UAE(self, encoder_net: Optional[keras.src.models.model.Model] = None, input_layer: Union[keras.src.layers.layer.Layer, keras.src.models.model.Model, NoneType] = None, shape: Optional[tuple] = None, enc_dim: Optional[int] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `encoder_net` | `Optional[keras.src.models.model.Model]` | `None` |  |
| `input_layer` | `Union[keras.src.layers.layer.Layer, keras.src.models.model.Model, None]` | `None` |  |
| `shape` | `Optional[tuple]` | `None` |  |
| `enc_dim` | `Optional[int]` | `None` |  |

### Methods

#### `call`

```python
call(x: Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor, Dict[str, tensorflow.python.framework.tensor.Tensor]]) -> tensorflow.python.framework.tensor.Tensor
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor, Dict[str, tensorflow.python.framework.tensor.Tensor]]` |  |  |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`

## Functions
### `preprocess_drift`

```python
preprocess_drift(x: Union[numpy.ndarray, list], model: keras.src.models.model.Model, preprocess_batch_fn: Optional[Callable] = None, tokenizer: Optional[Callable] = None, max_len: Optional[int] = None, batch_size: int = 10000000000, dtype: type[numpy.generic] = <class 'numpy.float32'>) -> Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor]
```

Prediction function used for preprocessing step of drift detector.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, list]` |  | Batch of instances. |
| `model` | `keras.src.models.model.Model` |  | Model used for preprocessing. |
| `preprocess_batch_fn` | `Optional[Callable]` | `None` | Optional batch preprocessing function. For example to convert a list of objects to a batch which can be processed by the TensorFlow model. |
| `tokenizer` | `Optional[Callable]` | `None` | Optional tokenizer for text drift. |
| `max_len` | `Optional[int]` | `None` | Optional max token length for text drift. |
| `batch_size` | `int` | `10000000000` | Batch size. |
| `dtype` | `type[numpy.generic]` | `<class 'numpy.float32'>` | Model output type, e.g. np.float32 or tf.float32. |

**Returns**
- Type: `Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor]`
