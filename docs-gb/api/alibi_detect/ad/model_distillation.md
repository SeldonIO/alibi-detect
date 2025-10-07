# `alibi_detect.ad.model_distillation`
## Constants
### `logger`
```python
logger: logging.Logger = <Logger alibi_detect.ad.model_distillation (WARNING)>
```
Instances of the Logger class represent a single logging channel. A
"logging channel" indicates an area of an application. Exactly how an
"area" is defined is up to the application developer. Since an
application can have any number of areas, logging channels are identified
by a unique string. Application areas can be nested (e.g. an area
of "input processing" might include sub-areas "read CSV files", "read
XLS files" and "read Gnumeric files"). To cater for this natural nesting,
channel names are organized into a namespace hierarchy where levels are
separated by periods, much like the Java or Python package namespace. So
in the instance given above, channel names might be "input" for the upper
level, and "input.csv", "input.xls" and "input.gnu" for the sub-levels.
There is no arbitrary limit to the depth of nesting.

## `ModelDistillation`

_Inherits from:_ `BaseDetector`, `FitMixin`, `ThresholdMixin`, `ABC`

### Constructor

```python
ModelDistillation(self, threshold: float = None, distilled_model: keras.src.models.model.Model = None, model: keras.src.models.model.Model = None, loss_type: str = 'kld', temperature: float = 1.0, data_type: str = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `threshold` | `Optional[float]` | `None` | Threshold used for score to determine adversarial instances. |
| `distilled_model` | `Optional[keras.src.models.model.Model]` | `None` | A tf.keras model to distill. |
| `model` | `Optional[keras.src.models.model.Model]` | `None` | A trained tf.keras classification model. |
| `loss_type` | `str` | `'kld'` | Loss for distillation. Supported: 'kld', 'xent' |
| `temperature` | `float` | `1.0` | Temperature used for model prediction scaling. Temperature <1 sharpens the prediction probability distribution. |
| `data_type` | `Optional[str]` | `None` | Optionally specifiy the data type (tabular, image or time-series). Added to metadata. |

### Methods

#### `fit`

```python
fit(X: numpy.ndarray, loss_fn: .tensorflow.keras.losses = <function loss_distillation at 0x28ee8c4c0>, optimizer: Union[ForwardRef('tf.keras.optimizers.Optimizer'), ForwardRef('tf.keras.optimizers.legacy.Optimizer'), type[ForwardRef('tf.keras.optimizers.Optimizer')], type[ForwardRef('tf.keras.optimizers.legacy.Optimizer')]] = <class 'keras.src.optimizers.adam.Adam'>, epochs: int = 20, batch_size: int = 128, verbose: bool = True, log_metric: Tuple[str, ForwardRef('tf.keras.metrics')] = None, callbacks: .tensorflow.keras.callbacks = None, preprocess_fn: Callable = None) -> None
```

Train ModelDistillation detector.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Training batch. |
| `loss_fn` | `.tensorflow.keras.losses` | `<function loss_distillation at 0x28ee8c4c0>` | Loss function used for training. |
| `optimizer` | `Union[ForwardRef('tf.keras.optimizers.Optimizer'), ForwardRef('tf.keras.optimizers.legacy.Optimizer'), type[ForwardRef('tf.keras.optimizers.Optimizer')], type[ForwardRef('tf.keras.optimizers.legacy.Optimizer')]]` | `<class 'keras.src.optimizers.adam.Adam'>` | Optimizer used for training. |
| `epochs` | `int` | `20` | Number of training epochs. |
| `batch_size` | `int` | `128` | Batch size used for training. |
| `verbose` | `bool` | `True` | Whether to print training progress. |
| `log_metric` | `Tuple[str, ForwardRef('tf.keras.metrics')]` | `None` | Additional metrics whose progress will be displayed if verbose equals True. |
| `callbacks` | `.tensorflow.keras.callbacks` | `None` | Callbacks used during training. |
| `preprocess_fn` | `Callable` | `None` | Preprocessing function applied to each training batch. |

**Returns**
- Type: `None`

#### `infer_threshold`

```python
infer_threshold(X: numpy.ndarray, threshold_perc: float = 99.0, margin: float = 0.0, batch_size: int = 10000000000) -> None
```

Update threshold by a value inferred from the percentage of instances considered to be

adversarial in a sample of the dataset.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Batch of instances. |
| `threshold_perc` | `float` | `99.0` | Percentage of X considered to be normal based on the adversarial score. |
| `margin` | `float` | `0.0` | Add margin to threshold. Useful if adversarial instances have significantly higher scores and there is no adversarial instance in X. |
| `batch_size` | `int` | `10000000000` | Batch size used when computing scores. |

**Returns**
- Type: `None`

#### `predict`

```python
predict(X: numpy.ndarray, batch_size: int = 10000000000, return_instance_score: bool = True) -> Dict[Dict[str, str], Dict[str, numpy.ndarray]]
```

Predict whether instances are adversarial instances or not.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Batch of instances. |
| `batch_size` | `int` | `10000000000` | Batch size used when computing scores. |
| `return_instance_score` | `bool` | `True` | Whether to return instance level adversarial scores. |

**Returns**
- Type: `Dict[Dict[str, str], Dict[str, numpy.ndarray]]`

#### `score`

```python
score(X: numpy.ndarray, batch_size: int = 10000000000, return_predictions: bool = False) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]]
```

Compute adversarial scores.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Batch of instances to analyze. |
| `batch_size` | `int` | `10000000000` | Batch size used when computing scores. |
| `return_predictions` | `bool` | `False` | Whether to return the predictions of the classifier on the original and reconstructed instances. |

**Returns**
- Type: `Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]]`
