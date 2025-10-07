# `alibi_detect.od.ae`
## Constants
### `logger`
```python
logger: logging.Logger = <Logger alibi_detect.od.ae (WARNING)>
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

## `OutlierAE`

_Inherits from:_ `BaseDetector`, `FitMixin`, `ThresholdMixin`, `ABC`

### Constructor

```python
OutlierAE(self, threshold: float = None, ae: keras.src.models.model.Model = None, encoder_net: keras.src.models.model.Model = None, decoder_net: keras.src.models.model.Model = None, data_type: str = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `threshold` | `Optional[float]` | `None` | Threshold used for outlier score to determine outliers. |
| `ae` | `Optional[keras.src.models.model.Model]` | `None` | A trained tf.keras model if available. |
| `encoder_net` | `Optional[keras.src.models.model.Model]` | `None` | Layers for the encoder wrapped in a tf.keras.Sequential class if no 'ae' is specified. |
| `decoder_net` | `Optional[keras.src.models.model.Model]` | `None` | Layers for the decoder wrapped in a tf.keras.Sequential class if no 'ae' is specified. |
| `data_type` | `Optional[str]` | `None` | Optionally specify the data type (tabular, image or time-series). Added to metadata. |

### Methods

#### `feature_score`

```python
feature_score(X_orig: numpy.ndarray, X_recon: numpy.ndarray) -> numpy.ndarray
```

Compute feature level outlier scores.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_orig` | `numpy.ndarray` |  | Batch of original instances. |
| `X_recon` | `numpy.ndarray` |  | Batch of reconstructed instances. |

**Returns**
- Type: `numpy.ndarray`

#### `fit`

```python
fit(X: numpy.ndarray, loss_fn: .tensorflow.keras.losses = <LossFunctionWrapper(<function mean_squared_error at 0x17511fca0>, kwargs={})>, optimizer: Union[ForwardRef('tf.keras.optimizers.Optimizer'), ForwardRef('tf.keras.optimizers.legacy.Optimizer'), type[ForwardRef('tf.keras.optimizers.Optimizer')], type[ForwardRef('tf.keras.optimizers.legacy.Optimizer')]] = <class 'keras.src.optimizers.adam.Adam'>, epochs: int = 20, batch_size: int = 64, verbose: bool = True, log_metric: Tuple[str, ForwardRef('tf.keras.metrics')] = None, callbacks: .tensorflow.keras.callbacks = None) -> None
```

Train AE model.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Training batch. |
| `loss_fn` | `.tensorflow.keras.losses` | `<LossFunctionWrapper(<function mean_squared_error at 0x17511fca0>, kwargs={})>` | Loss function used for training. |
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
infer_threshold(X: numpy.ndarray, outlier_type: str = 'instance', outlier_perc: float = 100.0, threshold_perc: float = 95.0, batch_size: int = 10000000000) -> None
```

Update threshold by a value inferred from the percentage of instances considered to be

outliers in a sample of the dataset.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Batch of instances. |
| `outlier_type` | `str` | `'instance'` | Predict outliers at the 'feature' or 'instance' level. |
| `outlier_perc` | `float` | `100.0` | Percentage of sorted feature level outlier scores used to predict instance level outlier. |
| `threshold_perc` | `float` | `95.0` | Percentage of X considered to be normal based on the outlier score. |
| `batch_size` | `int` | `10000000000` | Batch size used when making predictions with the autoencoder. |

**Returns**
- Type: `None`

#### `instance_score`

```python
instance_score(fscore: numpy.ndarray, outlier_perc: float = 100.0) -> numpy.ndarray
```

Compute instance level outlier scores.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `fscore` | `numpy.ndarray` |  | Feature level outlier scores. |
| `outlier_perc` | `float` | `100.0` | Percentage of sorted feature level outlier scores used to predict instance level outlier. |

**Returns**
- Type: `numpy.ndarray`

#### `predict`

```python
predict(X: numpy.ndarray, outlier_type: str = 'instance', outlier_perc: float = 100.0, batch_size: int = 10000000000, return_feature_score: bool = True, return_instance_score: bool = True) -> Dict[Dict[str, str], Dict[numpy.ndarray, numpy.ndarray]]
```

Predict whether instances are outliers or not.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Batch of instances. |
| `outlier_type` | `str` | `'instance'` | Predict outliers at the 'feature' or 'instance' level. |
| `outlier_perc` | `float` | `100.0` | Percentage of sorted feature level outlier scores used to predict instance level outlier. |
| `batch_size` | `int` | `10000000000` | Batch size used when making predictions with the autoencoder. |
| `return_feature_score` | `bool` | `True` | Whether to return feature level outlier scores. |
| `return_instance_score` | `bool` | `True` | Whether to return instance level outlier scores. |

**Returns**
- Type: `Dict[Dict[str, str], Dict[numpy.ndarray, numpy.ndarray]]`

#### `score`

```python
score(X: numpy.ndarray, outlier_perc: float = 100.0, batch_size: int = 10000000000) -> Tuple[numpy.ndarray, numpy.ndarray]
```

Compute feature and instance level outlier scores.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Batch of instances. |
| `outlier_perc` | `float` | `100.0` | Percentage of sorted feature level outlier scores used to predict instance level outlier. |
| `batch_size` | `int` | `10000000000` | Batch size used when making predictions with the autoencoder. |

**Returns**
- Type: `Tuple[numpy.ndarray, numpy.ndarray]`
