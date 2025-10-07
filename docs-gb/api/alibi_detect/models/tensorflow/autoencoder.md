# `alibi_detect.models.tensorflow.autoencoder`
## `AE`

_Inherits from:_ `Model`, `TensorFlowTrainer`, `Trainer`, `Layer`, `TFLayer`, `KerasAutoTrackable`, `AutoTrackable`, `Trackable`, `Operation`, `KerasSaveable`

### Constructor

```python
AE(self, encoder_net: keras.src.models.model.Model, decoder_net: keras.src.models.model.Model, name: str = 'ae') -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `encoder_net` | `keras.src.models.model.Model` |  | Layers for the encoder wrapped in a tf.keras.Sequential class. |
| `decoder_net` | `keras.src.models.model.Model` |  | Layers for the decoder wrapped in a tf.keras.Sequential class. |
| `name` | `str` | `'ae'` | Name of autoencoder model. |

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

## `AEGMM`

_Inherits from:_ `Model`, `TensorFlowTrainer`, `Trainer`, `Layer`, `TFLayer`, `KerasAutoTrackable`, `AutoTrackable`, `Trackable`, `Operation`, `KerasSaveable`

### Constructor

```python
AEGMM(self, encoder_net: keras.src.models.model.Model, decoder_net: keras.src.models.model.Model, gmm_density_net: keras.src.models.model.Model, n_gmm: int, recon_features: Callable = <function eucl_cosim_features at 0x282ff4430>, name: str = 'aegmm') -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `encoder_net` | `keras.src.models.model.Model` |  | Layers for the encoder wrapped in a tf.keras.Sequential class. |
| `decoder_net` | `keras.src.models.model.Model` |  | Layers for the decoder wrapped in a tf.keras.Sequential class. |
| `gmm_density_net` | `keras.src.models.model.Model` |  | Layers for the GMM network wrapped in a tf.keras.Sequential class. |
| `n_gmm` | `int` |  | Number of components in GMM. |
| `recon_features` | `Callable` | `<function eucl_cosim_features at 0x282ff4430>` | Function to extract features from the reconstructed instance by the decoder. |
| `name` | `str` | `'aegmm'` | Name of the AEGMM model. |

### Methods

#### `call`

```python
call(x: tensorflow.python.framework.tensor.Tensor) -> Tuple[tensorflow.python.framework.tensor.Tensor, tensorflow.python.framework.tensor.Tensor, tensorflow.python.framework.tensor.Tensor]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `tensorflow.python.framework.tensor.Tensor` |  |  |

**Returns**
- Type: `Tuple[tensorflow.python.framework.tensor.Tensor, tensorflow.python.framework.tensor.Tensor, tensorflow.python.framework.tensor.Tensor]`

## `Decoder`

_Inherits from:_ `Layer`, `TFLayer`, `KerasAutoTrackable`, `AutoTrackable`, `Trackable`, `Operation`, `KerasSaveable`

### Constructor

```python
Decoder(self, decoder_net: keras.src.models.model.Model, name: str = 'decoder') -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `decoder_net` | `keras.src.models.model.Model` |  | Layers for the decoder wrapped in a tf.keras.Sequential class. |
| `name` | `str` | `'decoder'` | Name of decoder. |

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

## `DecoderLSTM`

_Inherits from:_ `Layer`, `TFLayer`, `KerasAutoTrackable`, `AutoTrackable`, `Trackable`, `Operation`, `KerasSaveable`

### Constructor

```python
DecoderLSTM(self, latent_dim: int, output_dim: int, output_activation: str = None, name: str = 'decoder_lstm') -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `latent_dim` | `int` |  | Latent dimension. |
| `output_dim` | `int` |  | Decoder output dimension. |
| `output_activation` | `Optional[str]` | `None` | Activation used in the Dense output layer. |
| `name` | `str` | `'decoder_lstm'` | Name of decoder. |

### Methods

#### `call`

```python
call(x: tensorflow.python.framework.tensor.Tensor, init_state: List[tensorflow.python.framework.tensor.Tensor]) -> Tuple[tensorflow.python.framework.tensor.Tensor, tensorflow.python.framework.tensor.Tensor, List[tensorflow.python.framework.tensor.Tensor]]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `tensorflow.python.framework.tensor.Tensor` |  |  |
| `init_state` | `List[tensorflow.python.framework.tensor.Tensor]` |  |  |

**Returns**
- Type: `Tuple[tensorflow.python.framework.tensor.Tensor, tensorflow.python.framework.tensor.Tensor, List[tensorflow.python.framework.tensor.Tensor]]`

## `EncoderAE`

_Inherits from:_ `Layer`, `TFLayer`, `KerasAutoTrackable`, `AutoTrackable`, `Trackable`, `Operation`, `KerasSaveable`

### Constructor

```python
EncoderAE(self, encoder_net: keras.src.models.model.Model, name: str = 'encoder_ae') -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `encoder_net` | `keras.src.models.model.Model` |  | Layers for the encoder wrapped in a tf.keras.Sequential class. |
| `name` | `str` | `'encoder_ae'` | Name of encoder. |

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

## `EncoderLSTM`

_Inherits from:_ `Layer`, `TFLayer`, `KerasAutoTrackable`, `AutoTrackable`, `Trackable`, `Operation`, `KerasSaveable`

### Constructor

```python
EncoderLSTM(self, latent_dim: int, name: str = 'encoder_lstm') -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `latent_dim` | `int` |  | Latent dimension. Must be an even number given the bidirectional encoder. |
| `name` | `str` | `'encoder_lstm'` | Name of encoder. |

### Methods

#### `call`

```python
call(x: tensorflow.python.framework.tensor.Tensor) -> Tuple[tensorflow.python.framework.tensor.Tensor, List[tensorflow.python.framework.tensor.Tensor]]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `tensorflow.python.framework.tensor.Tensor` |  |  |

**Returns**
- Type: `Tuple[tensorflow.python.framework.tensor.Tensor, List[tensorflow.python.framework.tensor.Tensor]]`

## `EncoderVAE`

_Inherits from:_ `Layer`, `TFLayer`, `KerasAutoTrackable`, `AutoTrackable`, `Trackable`, `Operation`, `KerasSaveable`

### Constructor

```python
EncoderVAE(self, encoder_net: keras.src.models.model.Model, latent_dim: int, name: str = 'encoder_vae') -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `encoder_net` | `keras.src.models.model.Model` |  | Layers for the encoder wrapped in a tf.keras.Sequential class. |
| `latent_dim` | `int` |  | Dimensionality of the latent space. |
| `name` | `str` | `'encoder_vae'` | Name of encoder. |

### Methods

#### `call`

```python
call(x: tensorflow.python.framework.tensor.Tensor) -> Tuple[tensorflow.python.framework.tensor.Tensor, tensorflow.python.framework.tensor.Tensor, tensorflow.python.framework.tensor.Tensor]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `tensorflow.python.framework.tensor.Tensor` |  |  |

**Returns**
- Type: `Tuple[tensorflow.python.framework.tensor.Tensor, tensorflow.python.framework.tensor.Tensor, tensorflow.python.framework.tensor.Tensor]`

## `Sampling`

_Inherits from:_ `Layer`, `TFLayer`, `KerasAutoTrackable`, `AutoTrackable`, `Trackable`, `Operation`, `KerasSaveable`

Reparametrization trick. Uses (z_mean, z_log_var) to sample the latent vector z.

### Methods

#### `call`

```python
call(inputs: Tuple[tensorflow.python.framework.tensor.Tensor, tensorflow.python.framework.tensor.Tensor]) -> tensorflow.python.framework.tensor.Tensor
```

Sample z.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `inputs` | `Tuple[tensorflow.python.framework.tensor.Tensor, tensorflow.python.framework.tensor.Tensor]` |  | Tuple with mean and log variance. |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`

## `Seq2Seq`

_Inherits from:_ `Model`, `TensorFlowTrainer`, `Trainer`, `Layer`, `TFLayer`, `KerasAutoTrackable`, `AutoTrackable`, `Trackable`, `Operation`, `KerasSaveable`

### Constructor

```python
Seq2Seq(self, encoder_net: alibi_detect.models.tensorflow.autoencoder.EncoderLSTM, decoder_net: alibi_detect.models.tensorflow.autoencoder.DecoderLSTM, threshold_net: keras.src.models.model.Model, n_features: int, score_fn: Callable = <function squared_difference at 0x1237440d0>, beta: float = 1.0, name: str = 'seq2seq') -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `encoder_net` | `alibi_detect.models.tensorflow.autoencoder.EncoderLSTM` |  | Encoder network. |
| `decoder_net` | `alibi_detect.models.tensorflow.autoencoder.DecoderLSTM` |  | Decoder network. |
| `threshold_net` | `keras.src.models.model.Model` |  | Regression network used to estimate threshold. |
| `n_features` | `int` |  | Number of features. |
| `score_fn` | `Callable` | `<function squared_difference at 0x1237440d0>` | Function used for outlier score. |
| `beta` | `float` | `1.0` | Weight on the threshold estimation loss term. |
| `name` | `str` | `'seq2seq'` | Name of the seq2seq model. |

### Methods

#### `call`

```python
call(x: tensorflow.python.framework.tensor.Tensor) -> tensorflow.python.framework.tensor.Tensor
```

Forward pass used for teacher-forcing training.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `tensorflow.python.framework.tensor.Tensor` |  |  |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`

#### `decode_seq`

```python
decode_seq(x: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]
```

Sequence decoding and threshold estimation used for inference.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `numpy.ndarray` |  |  |

**Returns**
- Type: `Tuple[numpy.ndarray, numpy.ndarray]`

## `VAE`

_Inherits from:_ `Model`, `TensorFlowTrainer`, `Trainer`, `Layer`, `TFLayer`, `KerasAutoTrackable`, `AutoTrackable`, `Trackable`, `Operation`, `KerasSaveable`

### Constructor

```python
VAE(self, encoder_net: keras.src.models.model.Model, decoder_net: keras.src.models.model.Model, latent_dim: int, beta: float = 1.0, name: str = 'vae') -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `encoder_net` | `keras.src.models.model.Model` |  | Layers for the encoder wrapped in a tf.keras.Sequential class. |
| `decoder_net` | `keras.src.models.model.Model` |  | Layers for the decoder wrapped in a tf.keras.Sequential class. |
| `latent_dim` | `int` |  | Dimensionality of the latent space. |
| `beta` | `float` | `1.0` | Beta parameter for KL-divergence loss term. |
| `name` | `str` | `'vae'` | Name of VAE model. |

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

## `VAEGMM`

_Inherits from:_ `Model`, `TensorFlowTrainer`, `Trainer`, `Layer`, `TFLayer`, `KerasAutoTrackable`, `AutoTrackable`, `Trackable`, `Operation`, `KerasSaveable`

### Constructor

```python
VAEGMM(self, encoder_net: keras.src.models.model.Model, decoder_net: keras.src.models.model.Model, gmm_density_net: keras.src.models.model.Model, n_gmm: int, latent_dim: int, recon_features: Callable = <function eucl_cosim_features at 0x282ff4430>, beta: float = 1.0, name: str = 'vaegmm') -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `encoder_net` | `keras.src.models.model.Model` |  | Layers for the encoder wrapped in a tf.keras.Sequential class. |
| `decoder_net` | `keras.src.models.model.Model` |  | Layers for the decoder wrapped in a tf.keras.Sequential class. |
| `gmm_density_net` | `keras.src.models.model.Model` |  | Layers for the GMM network wrapped in a tf.keras.Sequential class. |
| `n_gmm` | `int` |  | Number of components in GMM. |
| `latent_dim` | `int` |  | Dimensionality of the latent space. |
| `recon_features` | `Callable` | `<function eucl_cosim_features at 0x282ff4430>` | Function to extract features from the reconstructed instance by the decoder. |
| `beta` | `float` | `1.0` | Beta parameter for KL-divergence loss term. |
| `name` | `str` | `'vaegmm'` | Name of the VAEGMM model. |

### Methods

#### `call`

```python
call(x: tensorflow.python.framework.tensor.Tensor) -> Tuple[tensorflow.python.framework.tensor.Tensor, tensorflow.python.framework.tensor.Tensor, tensorflow.python.framework.tensor.Tensor]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `tensorflow.python.framework.tensor.Tensor` |  |  |

**Returns**
- Type: `Tuple[tensorflow.python.framework.tensor.Tensor, tensorflow.python.framework.tensor.Tensor, tensorflow.python.framework.tensor.Tensor]`

## Functions
### `eucl_cosim_features`

```python
eucl_cosim_features(x: tensorflow.python.framework.tensor.Tensor, y: tensorflow.python.framework.tensor.Tensor, max_eucl: float = 100.0) -> tensorflow.python.framework.tensor.Tensor
```

Compute features extracted from the reconstructed instance using the

relative Euclidean distance and cosine similarity between 2 tensors.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `tensorflow.python.framework.tensor.Tensor` |  | Tensor used in feature computation. |
| `y` | `tensorflow.python.framework.tensor.Tensor` |  | Tensor used in feature computation. |
| `max_eucl` | `float` | `100.0` | Maximum value to clip relative Euclidean distance by. |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`
