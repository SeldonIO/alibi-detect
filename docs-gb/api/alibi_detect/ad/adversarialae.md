# `alibi_detect.ad.adversarialae`
## Constants
### `logger`
```python
logger: logging.Logger = <Logger alibi_detect.ad.adversarialae (WARNING)>
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

## `AdversarialAE`

_Inherits from:_ `BaseDetector`, `FitMixin`, `ThresholdMixin`, `ABC`

### Constructor

```python
AdversarialAE(self, threshold: float = None, ae: keras.src.models.model.Model = None, model: keras.src.models.model.Model = None, encoder_net: keras.src.models.model.Model = None, decoder_net: keras.src.models.model.Model = None, model_hl: List[keras.src.models.model.Model] = None, hidden_layer_kld: dict = None, w_model_hl: list = None, temperature: float = 1.0, data_type: str = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `threshold` | `Optional[float]` | `None` | Threshold used for adversarial score to determine adversarial instances. |
| `ae` | `Optional[keras.src.models.model.Model]` | `None` | A trained tf.keras autoencoder model if available. |
| `model` | `Optional[keras.src.models.model.Model]` | `None` | A trained tf.keras classification model. |
| `encoder_net` | `Optional[keras.src.models.model.Model]` | `None` | Layers for the encoder wrapped in a tf.keras.Sequential class if no 'ae' is specified. |
| `decoder_net` | `Optional[keras.src.models.model.Model]` | `None` | Layers for the decoder wrapped in a tf.keras.Sequential class if no 'ae' is specified. |
| `model_hl` | `Optional[List[keras.src.models.model.Model]]` | `None` | List with tf.keras models for the hidden layer K-L divergence computation. |
| `hidden_layer_kld` | `Optional[dict]` | `None` | Dictionary with as keys the hidden layer(s) of the model which are extracted and used during training of the AE, and as values the output dimension for the hidden layer. |
| `w_model_hl` | `Optional[list]` | `None` | Weights assigned to the loss of each model in model_hl. |
| `temperature` | `float` | `1.0` | Temperature used for model prediction scaling. Temperature <1 sharpens the prediction probability distribution. |
| `data_type` | `Optional[str]` | `None` | Optionally specifiy the data type (tabular, image or time-series). Added to metadata. |

### Methods

#### `correct`

```python
correct(X: numpy.ndarray, batch_size: int = 10000000000, return_instance_score: bool = True, return_all_predictions: bool = True) -> Dict[Dict[str, str], Dict[str, numpy.ndarray]]
```

Correct adversarial instances if the adversarial score is above the threshold.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Batch of instances. |
| `batch_size` | `int` | `10000000000` | Batch size used when computing scores. |
| `return_instance_score` | `bool` | `True` | Whether to return instance level adversarial scores. |
| `return_all_predictions` | `bool` | `True` | Whether to return the predictions on the original and the reconstructed data. |

**Returns**
- Type: `Dict[Dict[str, str], Dict[str, numpy.ndarray]]`

#### `fit`

```python
fit(X: numpy.ndarray, loss_fn: .tensorflow.keras.losses = <function loss_adv_ae at 0x290c4b3a0>, w_model: float = 1.0, w_recon: float = 0.0, optimizer: Union[ForwardRef('tf.keras.optimizers.Optimizer'), ForwardRef('tf.keras.optimizers.legacy.Optimizer'), type[ForwardRef('tf.keras.optimizers.Optimizer')], type[ForwardRef('tf.keras.optimizers.legacy.Optimizer')]] = <class 'keras.src.optimizers.adam.Adam'>, epochs: int = 20, batch_size: int = 128, verbose: bool = True, log_metric: Tuple[str, ForwardRef('tf.keras.metrics')] = None, callbacks: .tensorflow.keras.callbacks = None, preprocess_fn: Callable = None) -> None
```

Train Adversarial AE model.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Training batch. |
| `loss_fn` | `.tensorflow.keras.losses` | `<function loss_adv_ae at 0x290c4b3a0>` | Loss function used for training. |
| `w_model` | `float` | `1.0` | Weight on model prediction loss term. |
| `w_recon` | `float` | `0.0` | Weight on MSE reconstruction error loss term. |
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

## `DenseHidden`

_Inherits from:_ `Model`, `TensorFlowTrainer`, `Trainer`, `Layer`, `TFLayer`, `KerasAutoTrackable`, `AutoTrackable`, `Trackable`, `Operation`, `KerasSaveable`

### Constructor

```python
DenseHidden(self, model: keras.src.models.model.Model, hidden_layer: int, output_dim: int, hidden_dim: int = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `model` | `keras.src.models.model.Model` |  | tf.keras classification model. |
| `hidden_layer` | `int` |  | Hidden layer from model where feature map is extracted from. |
| `output_dim` | `int` |  | Output dimension for softmax layer. |
| `hidden_dim` | `Optional[int]` | `None` | Dimension of optional additional dense layer. |

### Methods

#### `call`

```python
call(x: tensorflow.python.framework.tensor.Tensor) -> tensorflow.python.framework.tensor.Tensor
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `tensorflow.python.framework.tensor.Tensor` |  |  |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`
