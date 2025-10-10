# `alibi_detect.models.tensorflow.trainer`
## Functions
### `trainer`

```python
trainer(model: keras.src.models.model.Model, loss_fn: .keras._tf_keras.keras.losses, x_train: numpy.ndarray, y_train: numpy.ndarray = None, dataset: keras.src.trainers.data_adapters.py_dataset_adapter.PyDataset = None, optimizer: .tensorflow.keras.optimizers = <class 'keras.src.optimizers.adam.Adam'>, loss_fn_kwargs: dict = None, preprocess_fn: Callable = None, epochs: int = 20, reg_loss_fn: Callable = <function <lambda> at 0x28ee82e50>, batch_size: int = 64, buffer_size: int = 1024, verbose: bool = True, log_metric: Tuple[str, ForwardRef('tf.keras.metrics')] = None, callbacks: .tensorflow.keras.callbacks = None) -> None
```

Train TensorFlow model.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `model` | `keras.src.models.model.Model` |  | Model to train. |
| `loss_fn` | `.keras._tf_keras.keras.losses` |  | Loss function used for training. |
| `x_train` | `numpy.ndarray` |  | Training data. |
| `y_train` | `numpy.ndarray` | `None` | Training labels. |
| `dataset` | `keras.src.trainers.data_adapters.py_dataset_adapter.PyDataset` | `None` | Training dataset which returns (x, y). |
| `optimizer` | `.tensorflow.keras.optimizers` | `<class 'keras.src.optimizers.adam.Adam'>` | Optimizer used for training. |
| `loss_fn_kwargs` | `dict` | `None` | Kwargs for loss function. |
| `preprocess_fn` | `Callable` | `None` | Preprocessing function applied to each training batch. |
| `epochs` | `int` | `20` | Number of training epochs. |
| `reg_loss_fn` | `Callable` | `<function <lambda> at 0x28ee82e50>` | Allows an additional regularisation term to be defined as reg_loss_fn(model) |
| `batch_size` | `int` | `64` | Batch size used for training. |
| `buffer_size` | `int` | `1024` | Maximum number of elements that will be buffered when prefetching. |
| `verbose` | `bool` | `True` | Whether to print training progress. |
| `log_metric` | `Tuple[str, ForwardRef('tf.keras.metrics')]` | `None` | Additional metrics whose progress will be displayed if verbose equals True. |
| `callbacks` | `.tensorflow.keras.callbacks` | `None` | Callbacks used during training. |

**Returns**
- Type: `None`
