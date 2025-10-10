# `alibi_detect.models.tensorflow.resnet`
## Constants
### `BATCH_NORM_DECAY`
```python
BATCH_NORM_DECAY: float = 0.997
```
Convert a string or number to a floating point number, if possible.

### `BATCH_NORM_EPSILON`
```python
BATCH_NORM_EPSILON: float = 1e-05
```
Convert a string or number to a floating point number, if possible.

### `L2_WEIGHT_DECAY`
```python
L2_WEIGHT_DECAY: float = 0.0002
```
Convert a string or number to a floating point number, if possible.

### `LR_SCHEDULE`
```python
LR_SCHEDULE: list = [(0.1, 91), (0.01, 136), (0.001, 182)]
```
Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### `BASE_LEARNING_RATE`
```python
BASE_LEARNING_RATE: float = 0.1
```
Convert a string or number to a floating point number, if possible.

### `HEIGHT`
```python
HEIGHT: int = 32
```
int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

### `WIDTH`
```python
WIDTH: int = 32
```
int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

### `NUM_CHANNELS`
```python
NUM_CHANNELS: int = 3
```
int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

## `LearningRateBatchScheduler`

_Inherits from:_ `Callback`

### Constructor

```python
LearningRateBatchScheduler(self, schedule: Callable, batch_size: int, steps_per_epoch: int)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `schedule` | `Callable` |  | Function taking the epoch and batch index as input which returns the new learning rate as output. |
| `batch_size` | `int` |  | Batch size. |
| `steps_per_epoch` | `int` |  | Number of batches or steps per epoch. |

### Methods

#### `on_batch_begin`

```python
on_batch_begin(batch, logs = None)
```

Executes before step begins.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `batch` |  |  |  |
| `logs` |  | `None` |  |

#### `on_epoch_begin`

```python
on_epoch_begin(epoch, logs = None)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `epoch` |  |  |  |
| `logs` |  | `None` |  |

## Functions
### `conv_block`

```python
conv_block(x_in: tensorflow.python.framework.tensor.Tensor, filters: Tuple[int, int], kernel_size: Union[int, list, Tuple[int]], stage: int, block: str, strides: Tuple[int, int] = (2, 2), l2_regularisation: bool = True) -> tensorflow.python.framework.tensor.Tensor
```

Conv block in ResNet with a parameterised skip connection to reduce the width and height

controlled by the strides.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_in` | `tensorflow.python.framework.tensor.Tensor` |  | Input Tensor. |
| `filters` | `Tuple[int, int]` |  | Number of filters for each of the 2 conv layers. |
| `kernel_size` | `Union[int, list, Tuple[int]]` |  | Kernel size for the conv layers. |
| `stage` | `int` |  | Stage of the block in the ResNet. |
| `block` | `str` |  | Block within a stage in the ResNet. |
| `strides` | `Tuple[int, int]` | `(2, 2)` | Stride size applied to reduce the image size. |
| `l2_regularisation` | `bool` | `True` | Whether to apply L2 regularisation. |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`

### `identity_block`

```python
identity_block(x_in: tensorflow.python.framework.tensor.Tensor, filters: Tuple[int, int], kernel_size: Union[int, list, Tuple[int]], stage: int, block: str, l2_regularisation: bool = True) -> tensorflow.python.framework.tensor.Tensor
```

Identity block in ResNet.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_in` | `tensorflow.python.framework.tensor.Tensor` |  | Input Tensor. |
| `filters` | `Tuple[int, int]` |  | Number of filters for each of the 2 conv layers. |
| `kernel_size` | `Union[int, list, Tuple[int]]` |  | Kernel size for the conv layers. |
| `stage` | `int` |  | Stage of the block in the ResNet. |
| `block` | `str` |  | Block within a stage in the ResNet. |
| `l2_regularisation` | `bool` | `True` | Whether to apply L2 regularisation. |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`

### `l2_regulariser`

```python
l2_regulariser(l2_regularisation: bool = True)
```

Apply L2 regularisation to kernel.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `l2_regularisation` | `bool` | `True` | Whether to apply L2 regularisation. |

### `learning_rate_schedule`

```python
learning_rate_schedule(current_epoch: int, current_batch: int, batches_per_epoch: int, batch_size: int) -> float
```

Linear learning rate scaling and learning rate decay at specified epochs.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `current_epoch` | `int` |  | Current training epoch. |
| `current_batch` | `int` |  | Current batch with current epoch, not used. |
| `batches_per_epoch` | `int` |  | Number of batches or steps in an epoch, not used. |
| `batch_size` | `int` |  | Batch size. |

**Returns**
- Type: `float`

### `preprocess_image`

```python
preprocess_image(x: numpy.ndarray, is_training: bool = True) -> numpy.ndarray
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `numpy.ndarray` |  |  |
| `is_training` | `bool` | `True` |  |

**Returns**
- Type: `numpy.ndarray`

### `resnet`

```python
resnet(num_blocks: int, classes: int = 10, input_shape: Tuple[int, int, int] = (32, 32, 3)) -> keras.src.models.model.Model
```

Define ResNet.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `num_blocks` | `int` |  | Number of ResNet blocks. |
| `classes` | `int` | `10` | Number of classification classes. |
| `input_shape` | `Tuple[int, int, int]` | `(32, 32, 3)` | Input shape of an image. |

**Returns**
- Type: `keras.src.models.model.Model`

### `resnet_block`

```python
resnet_block(x_in: tensorflow.python.framework.tensor.Tensor, size: int, filters: Tuple[int, int], kernel_size: Union[int, list, Tuple[int]], stage: int, strides: Tuple[int, int] = (2, 2), l2_regularisation: bool = True) -> tensorflow.python.framework.tensor.Tensor
```

Block in ResNet combining a conv block with identity blocks.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_in` | `tensorflow.python.framework.tensor.Tensor` |  | Input Tensor. |
| `size` | `int` |  | The ResNet block consists of 1 conv block and size-1 identity blocks. |
| `filters` | `Tuple[int, int]` |  | Number of filters for each of the conv layers. |
| `kernel_size` | `Union[int, list, Tuple[int]]` |  | Kernel size for the conv layers. |
| `stage` | `int` |  | Stage of the block in the ResNet. |
| `strides` | `Tuple[int, int]` | `(2, 2)` | Stride size applied to reduce the image size. |
| `l2_regularisation` | `bool` | `True` | Whether to apply L2 regularisation. |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`

### `run`

```python
run(num_blocks: int, epochs: int, batch_size: int, model_dir: Union[str, os.PathLike], num_classes: int = 10, input_shape: Tuple[int, int, int] = (32, 32, 3), validation_freq: int = 10, verbose: int = 2, seed: int = 1, serving: bool = False) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `num_blocks` | `int` |  |  |
| `epochs` | `int` |  |  |
| `batch_size` | `int` |  |  |
| `model_dir` | `Union[str, os.PathLike]` |  |  |
| `num_classes` | `int` | `10` |  |
| `input_shape` | `Tuple[int, int, int]` | `(32, 32, 3)` |  |
| `validation_freq` | `int` | `10` |  |
| `verbose` | `int` | `2` |  |
| `seed` | `int` | `1` |  |
| `serving` | `bool` | `False` |  |

**Returns**
- Type: `None`

### `scale_by_instance`

```python
scale_by_instance(x: numpy.ndarray, eps: float = 1e-12) -> numpy.ndarray
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `numpy.ndarray` |  |  |
| `eps` | `float` | `1e-12` |  |

**Returns**
- Type: `numpy.ndarray`
