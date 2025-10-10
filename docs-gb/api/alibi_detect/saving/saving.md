# `alibi_detect.saving.saving`
## Constants
### `TYPE_CHECKING`
```python
TYPE_CHECKING: bool = False
```
bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### `VALID_DETECTORS`
```python
VALID_DETECTORS: list = ['AdversarialAE', 'ChiSquareDrift', 'ClassifierDrift', 'IForest', 'KSDrift', ...
```
Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### `STATE_PATH`
```python
STATE_PATH: str = 'state/'
```
str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### `supported_models_all`
```python
supported_models_all: tuple = (<class 'keras.src.models.model.Model'>, <class 'torch.nn.modules.module.Modu...
```
Built-in immutable sequence.

If no argument is given, the constructor returns an empty tuple.
If iterable is specified the tuple is initialized from iterable's items.

If the argument is a tuple, the return value is the same object.

### `supported_models_tf`
```python
supported_models_tf: tuple = (<class 'keras.src.models.model.Model'>,)
```
Built-in immutable sequence.

If no argument is given, the constructor returns an empty tuple.
If iterable is specified the tuple is initialized from iterable's items.

If the argument is a tuple, the return value is the same object.

### `supported_models_torch`
```python
supported_models_torch: tuple = (<class 'torch.nn.modules.module.Module'>,)
```
Built-in immutable sequence.

If no argument is given, the constructor returns an empty tuple.
If iterable is specified the tuple is initialized from iterable's items.

If the argument is a tuple, the return value is the same object.

### `supported_models_sklearn`
```python
supported_models_sklearn: tuple = (<class 'sklearn.base.BaseEstimator'>,)
```
Built-in immutable sequence.

If no argument is given, the constructor returns an empty tuple.
If iterable is specified the tuple is initialized from iterable's items.

If the argument is a tuple, the return value is the same object.

### `logger`
```python
logger: logging.Logger = <Logger alibi_detect.saving.saving (WARNING)>
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

### `X_REF_FILENAME`
```python
X_REF_FILENAME: str = 'x_ref.npy'
```
str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### `C_REF_FILENAME`
```python
C_REF_FILENAME: str = 'c_ref.npy'
```
str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

## Functions
### `save_detector`

```python
save_detector(detector: Union[alibi_detect.base.Detector, alibi_detect.base.ConfigurableDetector], filepath: Union[str, os.PathLike], legacy: bool = False) -> None
```

Save outlier, drift or adversarial detector.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `detector` | `Union[alibi_detect.base.Detector, alibi_detect.base.ConfigurableDetector]` |  | Detector object. |
| `filepath` | `Union[str, os.PathLike]` |  | Save directory. |
| `legacy` | `bool` | `False` | Whether to save in the legacy .dill format instead of via a config.toml file. Default is `False`. This option will be removed in a future version. |

**Returns**
- Type: `None`

### `write_config`

```python
write_config(cfg: dict, filepath: Union[str, os.PathLike])
```

Save an unresolved detector config dict to a TOML file.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `cfg` | `dict` |  | Unresolved detector config dict. |
| `filepath` | `Union[str, os.PathLike]` |  | Filepath to directory to save 'config.toml' file in. |
