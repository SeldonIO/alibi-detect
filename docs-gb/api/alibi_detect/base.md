# `alibi_detect.base`
## Constants
### `DEFAULT_META`
```python
DEFAULT_META: dict = {'name': None, 'online': None, 'data_type': None, 'version': None, 'detector_...
```

### `LARGE_ARTEFACTS`
```python
LARGE_ARTEFACTS: list = ['x_ref', 'c_ref', 'preprocess_fn']
```
Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

## `BaseDetector`

_Inherits from:_ `ABC`

Base class for outlier, adversarial and drift detection algorithms.

### Constructor

```python
BaseDetector(self)
```
### Properties

| Property | Type | Description |
| -------- | ---- | ----------- |
| `meta` | `Dict` |  |

### Methods

#### `predict`

```python
predict(X: numpy.ndarray)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |

#### `score`

```python
score(X: numpy.ndarray)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |

## `ConfigurableDetector`

_Inherits from:_ `Detector`, `Protocol`, `Generic`

Type Protocol for detectors that have support for saving via config.

Used for typing save and load functionality in `alibi_detect.saving.saving`.

### Methods

#### `from_config`

```python
from_config(config: dict)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `config` | `dict` |  |  |

#### `get_config`

```python
get_config() -> dict
```

**Returns**
- Type: `dict`

## `Detector`

_Inherits from:_ `Protocol`, `Generic`

Type Protocol for all detectors.

Used for typing legacy save and load functionality in `alibi_detect.saving._tensorflow.saving.py`.

Note
----
    This exists to distinguish between detectors with and without support for config saving and loading. Once all
    detector support this then this protocol will be removed.

### Constructor

```python
Detector(self, *args, **kwargs)
```
### Methods

#### `predict`

```python
predict() -> typing.Any
```

**Returns**
- Type: `typing.Any`

## `DriftConfigMixin`

A mixin class containing methods related to a drift detector's configuration dictionary.

### Constructor

```python
DriftConfigMixin(self, /, *args, **kwargs)
```
### Methods

#### `from_config`

```python
from_config(config: dict)
```

Instantiate a drift detector from a fully resolved (and validated) config dictionary.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `config` | `dict` |  | A config dictionary matching the schema's in :class:`~alibi_detect.saving.schemas`. |

#### `get_config`

```python
get_config() -> dict
```

Get the detector's configuration dictionary.

**Returns**
- Type: `dict`

## `FitMixin`

_Inherits from:_ `ABC`

### Methods

#### `fit`

```python
fit(args, kwargs) -> None
```

**Returns**
- Type: `None`

## `NumpyEncoder`

_Inherits from:_ `JSONEncoder`

### Methods

#### `default`

```python
default(obj)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `obj` |  |  |  |

## `StatefulDetectorOnline`

_Inherits from:_ `ConfigurableDetector`, `Detector`, `Protocol`, `Generic`

Type Protocol for detectors that have support for save/loading of online state.

Used for typing save and load functionality in `alibi_detect.saving.saving`.

### Methods

#### `load_state`

```python
load_state(filepath: Union[str, os.PathLike])
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `filepath` | `Union[str, os.PathLike]` |  |  |

#### `save_state`

```python
save_state(filepath: Union[str, os.PathLike])
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `filepath` | `Union[str, os.PathLike]` |  |  |

## `ThresholdMixin`

_Inherits from:_ `ABC`

### Methods

#### `infer_threshold`

```python
infer_threshold(args, kwargs) -> None
```

**Returns**
- Type: `None`

## Functions
### `adversarial_correction_dict`

```python
adversarial_correction_dict()
```

### `adversarial_prediction_dict`

```python
adversarial_prediction_dict()
```

### `concept_drift_dict`

```python
concept_drift_dict()
```

### `outlier_prediction_dict`

```python
outlier_prediction_dict()
```
