# `alibi_detect.saving.loading`
## Constants
### `TYPE_CHECKING`
```python
TYPE_CHECKING: bool = False
```
bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### `has_tensorflow`
```python
has_tensorflow: bool = True
```
bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### `has_pytorch`
```python
has_pytorch: bool = True
```
bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

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

### `logger`
```python
logger: logging.Logger = <Logger alibi_detect.saving.loading (WARNING)>
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

### `FIELDS_TO_RESOLVE`
```python
FIELDS_TO_RESOLVE: list = [['preprocess_fn', 'src'], ['preprocess_fn', 'model'], ['preprocess_fn', 'emb...
```
Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### `FIELDS_TO_DTYPE`
```python
FIELDS_TO_DTYPE: list = [['preprocess_fn', 'dtype']]
```
Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

## Functions
### `load_detector`

```python
load_detector(filepath: Union[str, os.PathLike], enable_unsafe_loading: bool = False, kwargs) -> Union[alibi_detect.base.Detector, alibi_detect.base.ConfigurableDetector]
```

Load outlier, drift or adversarial detector.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `filepath` | `Union[str, os.PathLike]` |  | Load directory. |
| `enable_unsafe_loading` | `bool` | `False` | Sets allow_pickle=True when a np.ndarray is loaded from a .npy file referenced in the detector config. Needed if you have to load objects. Only applied if the filepath is config.toml or a directory containing a config.toml. It has security implications: https://nvd.nist.gov/vuln/detail/cve-2019-6446. |

**Returns**
- Type: `Union[alibi_detect.base.Detector, alibi_detect.base.ConfigurableDetector]`

### `read_config`

```python
read_config(filepath: Union[os.PathLike, str]) -> dict
```

This function reads a detector toml config file and returns a dict specifying the detector.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `filepath` | `Union[os.PathLike, str]` |  | The filepath to the config.toml file. |

**Returns**
- Type: `dict`

### `resolve_config`

```python
resolve_config(cfg: dict, config_dir: Optional[pathlib.Path], enable_unsafe_loading: bool = False) -> dict
```

Resolves artefacts in a config dict. For example x_ref='x_ref.npy' is resolved by loading the np.ndarray from

the .npy file. For a list of fields that are resolved, see
https://docs.seldon.io/projects/alibi-detect/en/stable/overview/config_file.html.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `cfg` | `dict` |  | The unresolved config dict. |
| `config_dir` | `Optional[pathlib.Path]` |  | Filepath to directory the `config.toml` is located in. Only required if different from the runtime directory, and artefacts are specified with filepaths relative to the config.toml file. |
| `enable_unsafe_loading` | `bool` | `False` | If set to true, allow_pickle=True is set in np.load(). Needed if you have to load objects. It has security implications: https://nvd.nist.gov/vuln/detail/cve-2019-6446 |

**Returns**
- Type: `dict`
