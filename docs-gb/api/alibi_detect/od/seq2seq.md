# `alibi_detect.od.seq2seq`
## Constants
### `logger`
```python
logger: logging.Logger = <Logger alibi_detect.od.seq2seq (WARNING)>
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

## `OutlierSeq2Seq`

_Inherits from:_ `BaseDetector`, `FitMixin`, `ThresholdMixin`, `ABC`

### Constructor

```python
OutlierSeq2Seq(self, n_features: int, seq_len: int, threshold: Union[float, numpy.ndarray] = None, seq2seq: keras.src.models.model.Model = None, threshold_net: keras.src.models.model.Model = None, latent_dim: int = None, output_activation: str = None, beta: float = 1.0) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `n_features` | `int` |  | Number of features in the time series. |
| `seq_len` | `int` |  | Sequence length fed into the Seq2Seq model. |
| `threshold` | `Union[float, numpy.ndarray, None]` | `None` | Threshold used for outlier detection. Can be a float or feature-wise array. |
| `seq2seq` | `Optional[keras.src.models.model.Model]` | `None` | A trained seq2seq model if available. |
| `threshold_net` | `Optional[keras.src.models.model.Model]` | `None` | Layers for the threshold estimation network wrapped in a tf.keras.Sequential class if no 'seq2seq' is specified. |
| `latent_dim` | `Optional[int]` | `None` | Latent dimension of the encoder and decoder. |
| `output_activation` | `Optional[str]` | `None` | Activation used in the Dense output layer of the decoder. |
| `beta` | `float` | `1.0` | Weight on the threshold estimation loss term. |

### Methods

#### `feature_score`

```python
feature_score(X_orig: numpy.ndarray, X_recon: numpy.ndarray, threshold_est: numpy.ndarray) -> numpy.ndarray
```

Compute feature level outlier scores.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_orig` | `numpy.ndarray` |  | Original time series. |
| `X_recon` | `numpy.ndarray` |  | Reconstructed time series. |
| `threshold_est` | `numpy.ndarray` |  | Estimated threshold from the decoder's latent space. |

**Returns**
- Type: `numpy.ndarray`

#### `fit`

```python
fit(X: numpy.ndarray, loss_fn: .tensorflow.keras.losses = <function mean_squared_error at 0x169a64ca0>, optimizer: Union[ForwardRef('tf.keras.optimizers.Optimizer'), ForwardRef('tf.keras.optimizers.legacy.Optimizer'), type[ForwardRef('tf.keras.optimizers.Optimizer')], type[ForwardRef('tf.keras.optimizers.legacy.Optimizer')]] = <class 'keras.src.optimizers.adam.Adam'>, epochs: int = 20, batch_size: int = 64, verbose: bool = True, log_metric: Tuple[str, ForwardRef('tf.keras.metrics')] = None, callbacks: .tensorflow.keras.callbacks = None) -> None
```

Train Seq2Seq model.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Univariate or multivariate time series. Shape equals (batch, features) or (batch, sequence length, features). |
| `loss_fn` | `.tensorflow.keras.losses` | `<function mean_squared_error at 0x169a64ca0>` | Loss function used for training. |
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
infer_threshold(X: numpy.ndarray, outlier_perc: Union[int, float] = 100.0, threshold_perc: Union[int, float, numpy.ndarray, list] = 95.0, batch_size: int = 10000000000) -> None
```

Update the outlier threshold by using a sequence of instances from the dataset

of which the fraction of features which are outliers are known. This fraction can be across
all features or per feature.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Univariate or multivariate time series. |
| `outlier_perc` | `Union[int, float]` | `100.0` | Percentage of sorted feature level outlier scores used to predict instance level outlier. |
| `threshold_perc` | `Union[int, float, numpy.ndarray, list]` | `95.0` | Percentage of X considered to be normal based on the outlier score. Overall (float) or feature-wise (array or list). |
| `batch_size` | `int` | `10000000000` | Batch size used when making predictions with the seq2seq model. |

**Returns**
- Type: `None`

#### `instance_score`

```python
instance_score(fscore: numpy.ndarray, outlier_perc: float = 100.0) -> numpy.ndarray
```

Compute instance level outlier scores. `instance` in this case means the data along the

first axis of the original time series passed to the predictor.

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

Compute outlier scores and transform into outlier predictions.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Univariate or multivariate time series. |
| `outlier_type` | `str` | `'instance'` | Predict outliers at the 'feature' or 'instance' level. |
| `outlier_perc` | `float` | `100.0` | Percentage of sorted feature level outlier scores used to predict instance level outlier. |
| `batch_size` | `int` | `10000000000` | Batch size used when making predictions with the seq2seq model. |
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
| `X` | `numpy.ndarray` |  | Univariate or multivariate time series. |
| `outlier_perc` | `float` | `100.0` | Percentage of sorted feature level outlier scores used to predict instance level outlier. |
| `batch_size` | `int` | `10000000000` | Batch size used when making predictions with the seq2seq model. |

**Returns**
- Type: `Tuple[numpy.ndarray, numpy.ndarray]`
