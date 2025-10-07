# `alibi_detect.od.llr`
## Constants
### `logger`
```python
logger: logging.Logger = <Logger alibi_detect.od.llr (WARNING)>
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

## `LLR`

_Inherits from:_ `BaseDetector`, `FitMixin`, `ThresholdMixin`, `ABC`

### Constructor

```python
LLR(self, threshold: float = None, model: Union[keras.src.models.model.Model, tensorflow_probability.python.distributions.distribution.Distribution, alibi_detect.models.tensorflow.pixelcnn.PixelCNN] = None, model_background: Union[keras.src.models.model.Model, tensorflow_probability.python.distributions.distribution.Distribution, alibi_detect.models.tensorflow.pixelcnn.PixelCNN] = None, log_prob: Callable = None, sequential: bool = False, data_type: str = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `threshold` | `Optional[float]` | `None` | Threshold used for the likelihood ratio (LLR) to determine outliers. |
| `model` | `Union[keras.src.models.model.Model, tensorflow_probability.python.distributions.distribution.Distribution, alibi_detect.models.tensorflow.pixelcnn.PixelCNN, None]` | `None` | Generative model, defaults to PixelCNN. |
| `model_background` | `Union[keras.src.models.model.Model, tensorflow_probability.python.distributions.distribution.Distribution, alibi_detect.models.tensorflow.pixelcnn.PixelCNN, None]` | `None` | Optional model for the background. Only needed if it is different from `model`. |
| `log_prob` | `Optional[Callable]` | `None` | Function used to evaluate log probabilities under the model if the model does not have a `log_prob` function. |
| `sequential` | `bool` | `False` | Whether the data is sequential. Used to create targets during training. |
| `data_type` | `Optional[str]` | `None` | Optionally specify the data type (tabular, image or time-series). Added to metadata. |

### Methods

#### `feature_score`

```python
feature_score(X: numpy.ndarray, batch_size: int = 10000000000) -> numpy.ndarray
```

Feature-level negative likelihood ratios.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `batch_size` | `int` | `10000000000` |  |

**Returns**
- Type: `numpy.ndarray`

#### `fit`

```python
fit(X: numpy.ndarray, mutate_fn: Callable = <function mutate_categorical at 0x16b9a9a60>, mutate_fn_kwargs: dict = {'rate': 0.2, 'seed': 0, 'feature_range': (0, 255)}, mutate_batch_size: int = 10000000000, loss_fn: .tensorflow.keras.losses = None, loss_fn_kwargs: dict = None, optimizer: Union[ForwardRef('tf.keras.optimizers.Optimizer'), ForwardRef('tf.keras.optimizers.legacy.Optimizer'), type[ForwardRef('tf.keras.optimizers.Optimizer')], type[ForwardRef('tf.keras.optimizers.legacy.Optimizer')]] = <class 'keras.src.optimizers.adam.Adam'>, epochs: int = 20, batch_size: int = 64, verbose: bool = True, log_metric: Tuple[str, ForwardRef('tf.keras.metrics')] = None, callbacks: .tensorflow.keras.callbacks = None) -> None
```

Train semantic and background generative models.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Training batch. |
| `mutate_fn` | `Callable` | `<function mutate_categorical at 0x16b9a9a60>` | Mutation function used to generate the background dataset. |
| `mutate_fn_kwargs` | `dict` | `{'rate': 0.2, 'seed': 0, 'feature_range': (0, 255)}` | Kwargs for the mutation function used to generate the background dataset. Default values set for an image dataset. |
| `mutate_batch_size` | `int` | `10000000000` | Batch size used to generate the mutations for the background dataset. |
| `loss_fn` | `.tensorflow.keras.losses` | `None` | Loss function used for training. |
| `loss_fn_kwargs` | `dict` | `None` | Kwargs for loss function. |
| `optimizer` | `Union[ForwardRef('tf.keras.optimizers.Optimizer'), ForwardRef('tf.keras.optimizers.legacy.Optimizer'), type[ForwardRef('tf.keras.optimizers.Optimizer')], type[ForwardRef('tf.keras.optimizers.legacy.Optimizer')]]` | `<class 'keras.src.optimizers.adam.Adam'>` | Optimizer used for training. |
| `epochs` | `int` | `20` | Number of training epochs. |
| `batch_size` | `int` | `64` | Batch size used for training. |
| `verbose` | `bool` | `True` | Whether to print training progress. |
| `log_metric` | `Tuple[str, ForwardRef('tf.keras.metrics')]` | `None` | Additional metrics whose progress will be displayed if verbose equals True. |
| `callbacks` | `.tensorflow.keras.callbacks` | `None` | Callbacks used during training. |

**Returns**
- Type: `None`

#### `infer_threshold`

```python
infer_threshold(X: numpy.ndarray, outlier_type: str = 'instance', threshold_perc: float = 95.0, batch_size: int = 10000000000) -> None
```

Update LLR threshold by a value inferred from the percentage of instances

considered to be outliers in a sample of the dataset.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Batch of instances. |
| `outlier_type` | `str` | `'instance'` | Predict outliers at the 'feature' or 'instance' level. |
| `threshold_perc` | `float` | `95.0` | Percentage of sorted feature level outlier scores used to predict instance level outlier. |
| `batch_size` | `int` | `10000000000` | Batch size for the generative model evaluations. |

**Returns**
- Type: `None`

#### `instance_score`

```python
instance_score(X: numpy.ndarray, batch_size: int = 10000000000) -> numpy.ndarray
```

Instance-level negative likelihood ratios.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `batch_size` | `int` | `10000000000` |  |

**Returns**
- Type: `numpy.ndarray`

#### `llr`

```python
llr(X: numpy.ndarray, return_per_feature: bool, batch_size: int = 10000000000) -> numpy.ndarray
```

Compute likelihood ratios.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Batch of instances. |
| `return_per_feature` | `bool` |  | Return likelihood ratio per feature. |
| `batch_size` | `int` | `10000000000` | Batch size for the generative model evaluations. |

**Returns**
- Type: `numpy.ndarray`

#### `logp`

```python
logp(dist, X: numpy.ndarray, return_per_feature: bool = False, batch_size: int = 10000000000) -> numpy.ndarray
```

Compute log probability of a batch of instances under the generative model.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `dist` |  |  | Distribution of the model. |
| `X` | `numpy.ndarray` |  | Batch of instances. |
| `return_per_feature` | `bool` | `False` | Return log probability per feature. |
| `batch_size` | `int` | `10000000000` | Batch size for the generative model evaluations. |

**Returns**
- Type: `numpy.ndarray`

#### `logp_alt`

```python
logp_alt(model: keras.src.models.model.Model, X: numpy.ndarray, return_per_feature: bool = False, batch_size: int = 10000000000) -> numpy.ndarray
```

Compute log probability of a batch of instances using the log_prob function

defined by the user.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `model` | `keras.src.models.model.Model` |  | Trained model. |
| `X` | `numpy.ndarray` |  | Batch of instances. |
| `return_per_feature` | `bool` | `False` | Return log probability per feature. |
| `batch_size` | `int` | `10000000000` | Batch size for the generative model evaluations. |

**Returns**
- Type: `numpy.ndarray`

#### `predict`

```python
predict(X: numpy.ndarray, outlier_type: str = 'instance', batch_size: int = 10000000000, return_feature_score: bool = True, return_instance_score: bool = True) -> Dict[Dict[str, str], Dict[numpy.ndarray, numpy.ndarray]]
```

Predict whether instances are outliers or not.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Batch of instances. |
| `outlier_type` | `str` | `'instance'` | Predict outliers at the 'feature' or 'instance' level. |
| `batch_size` | `int` | `10000000000` | Batch size used when making predictions with the generative model. |
| `return_feature_score` | `bool` | `True` | Whether to return feature level outlier scores. |
| `return_instance_score` | `bool` | `True` | Whether to return instance level outlier scores. |

**Returns**
- Type: `Dict[Dict[str, str], Dict[numpy.ndarray, numpy.ndarray]]`

#### `score`

```python
score(X: numpy.ndarray, batch_size: int = 10000000000) -> Tuple[numpy.ndarray, numpy.ndarray]
```

Feature-level and instance-level outlier scores.

The scores are equal to the negative likelihood ratios.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `batch_size` | `int` | `10000000000` |  |

**Returns**
- Type: `Tuple[numpy.ndarray, numpy.ndarray]`

## Functions
### `build_model`

```python
build_model(dist: Union[tensorflow_probability.python.distributions.distribution.Distribution, alibi_detect.models.tensorflow.pixelcnn.PixelCNN], input_shape: Optional[tuple] = None, filepath: Optional[str] = None) -> Tuple[keras.src.models.model.Model, Union[tensorflow_probability.python.distributions.distribution.Distribution, alibi_detect.models.tensorflow.pixelcnn.PixelCNN]]
```

Create tf.keras.Model from TF distribution.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `dist` | `Union[tensorflow_probability.python.distributions.distribution.Distribution, alibi_detect.models.tensorflow.pixelcnn.PixelCNN]` |  | TensorFlow distribution. |
| `input_shape` | `Optional[tuple]` | `None` | Input shape of the model. |
| `filepath` | `Optional[str]` | `None` | File to load model weights from. |

**Returns**
- Type: `Tuple[keras.src.models.model.Model, Union[tensorflow_probability.python.distributions.distribution.Distribution, alibi_detect.models.tensorflow.pixelcnn.PixelCNN]]`
