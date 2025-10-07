# `alibi_detect.od.aegmm`
## Constants
### `logger`
```python
logger: logging.Logger = <Logger alibi_detect.od.aegmm (WARNING)>
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

## `OutlierAEGMM`

_Inherits from:_ `BaseDetector`, `FitMixin`, `ThresholdMixin`, `ABC`

### Constructor

```python
OutlierAEGMM(self, threshold: float = None, aegmm: keras.src.models.model.Model = None, encoder_net: keras.src.models.model.Model = None, decoder_net: keras.src.models.model.Model = None, gmm_density_net: keras.src.models.model.Model = None, n_gmm: int = None, recon_features: Callable = <function eucl_cosim_features at 0x282ff4430>, data_type: str = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `threshold` | `Optional[float]` | `None` | Threshold used for outlier score to determine outliers. |
| `aegmm` | `Optional[keras.src.models.model.Model]` | `None` | A trained tf.keras model if available. |
| `encoder_net` | `Optional[keras.src.models.model.Model]` | `None` | Layers for the encoder wrapped in a tf.keras.Sequential class if no 'aegmm' is specified. |
| `decoder_net` | `Optional[keras.src.models.model.Model]` | `None` | Layers for the decoder wrapped in a tf.keras.Sequential class if no 'aegmm' is specified. |
| `gmm_density_net` | `Optional[keras.src.models.model.Model]` | `None` | Layers for the GMM network wrapped in a tf.keras.Sequential class. |
| `n_gmm` | `Optional[int]` | `None` | Number of components in GMM. |
| `recon_features` | `Callable` | `<function eucl_cosim_features at 0x282ff4430>` | Function to extract features from the reconstructed instance by the decoder. |
| `data_type` | `Optional[str]` | `None` | Optionally specifiy the data type (tabular, image or time-series). Added to metadata. |

### Methods

#### `fit`

```python
fit(X: numpy.ndarray, loss_fn: .tensorflow.keras.losses = <function loss_aegmm at 0x290c4b280>, w_energy: float = 0.1, w_cov_diag: float = 0.005, optimizer: Union[ForwardRef('tf.keras.optimizers.Optimizer'), ForwardRef('tf.keras.optimizers.legacy.Optimizer'), type[ForwardRef('tf.keras.optimizers.Optimizer')], type[ForwardRef('tf.keras.optimizers.legacy.Optimizer')]] = <class 'keras.src.optimizers.adam.Adam'>, epochs: int = 20, batch_size: int = 64, verbose: bool = True, log_metric: Tuple[str, ForwardRef('tf.keras.metrics')] = None, callbacks: .tensorflow.keras.callbacks = None) -> None
```

Train AEGMM model.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Training batch. |
| `loss_fn` | `.tensorflow.keras.losses` | `<function loss_aegmm at 0x290c4b280>` | Loss function used for training. |
| `w_energy` | `float` | `0.1` | Weight on sample energy loss term if default `loss_aegmm` loss fn is used. |
| `w_cov_diag` | `float` | `0.005` | Weight on covariance regularizing loss term if default `loss_aegmm` loss fn is used. |
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
infer_threshold(X: numpy.ndarray, threshold_perc: float = 95.0, batch_size: int = 10000000000) -> None
```

Update threshold by a value inferred from the percentage of instances considered to be

outliers in a sample of the dataset.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Batch of instances. |
| `threshold_perc` | `float` | `95.0` | Percentage of X considered to be normal based on the outlier score. |
| `batch_size` | `int` | `10000000000` | Batch size used when making predictions with the AEGMM. |

**Returns**
- Type: `None`

#### `predict`

```python
predict(X: numpy.ndarray, batch_size: int = 10000000000, return_instance_score: bool = True) -> Dict[Dict[str, str], Dict[numpy.ndarray, numpy.ndarray]]
```

Compute outlier scores and transform into outlier predictions.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Batch of instances. |
| `batch_size` | `int` | `10000000000` | Batch size used when making predictions with the AEGMM. |
| `return_instance_score` | `bool` | `True` | Whether to return instance level outlier scores. |

**Returns**
- Type: `Dict[Dict[str, str], Dict[numpy.ndarray, numpy.ndarray]]`

#### `score`

```python
score(X: numpy.ndarray, batch_size: int = 10000000000) -> numpy.ndarray
```

Compute outlier scores.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Batch of instances to analyze. |
| `batch_size` | `int` | `10000000000` | Batch size used when making predictions with the AEGMM. |

**Returns**
- Type: `numpy.ndarray`
