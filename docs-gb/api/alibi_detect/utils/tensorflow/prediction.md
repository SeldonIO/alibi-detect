# `alibi_detect.utils.tensorflow.prediction`
## Functions
### `predict_batch`

```python
predict_batch(x: Union[list, numpy.ndarray, tensorflow.python.framework.tensor.Tensor], model: Union[Callable, keras.src.models.model.Model], batch_size: int = 10000000000, preprocess_fn: Optional[Callable] = None, dtype: Union[type[numpy.generic], tensorflow.python.framework.dtypes.DType] = <class 'numpy.float32'>) -> Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor, tuple]
```

Make batch predictions on a model.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[list, numpy.ndarray, tensorflow.python.framework.tensor.Tensor]` |  | Batch of instances. |
| `model` | `Union[Callable, keras.src.models.model.Model]` |  | tf.keras model or one of the other permitted types defined in Data. |
| `batch_size` | `int` | `10000000000` | Batch size used during prediction. |
| `preprocess_fn` | `Optional[Callable]` | `None` | Optional preprocessing function for each batch. |
| `dtype` | `Union[type[numpy.generic], tensorflow.python.framework.dtypes.DType]` | `<class 'numpy.float32'>` | Model output type, e.g. np.float32 or tf.float32. |

**Returns**
- Type: `Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor, tuple]`

### `predict_batch_transformer`

```python
predict_batch_transformer(x: Union[list, numpy.ndarray], model: keras.src.models.model.Model, tokenizer: Callable, max_len: int, batch_size: int = 10000000000, dtype: Union[type[numpy.generic], tensorflow.python.framework.dtypes.DType] = <class 'numpy.float32'>) -> Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor]
```

Make batch predictions using a transformers tokenizer and model.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[list, numpy.ndarray]` |  | Batch of instances. |
| `model` | `keras.src.models.model.Model` |  | Transformer model. |
| `tokenizer` | `Callable` |  | Tokenizer for model. |
| `max_len` | `int` |  | Max token length. |
| `batch_size` | `int` | `10000000000` | Batch size. |
| `dtype` | `Union[type[numpy.generic], tensorflow.python.framework.dtypes.DType]` | `<class 'numpy.float32'>` | Model output type, e.g. np.float32 or tf.float32. |

**Returns**
- Type: `Union[numpy.ndarray, tensorflow.python.framework.tensor.Tensor]`
