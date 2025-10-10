# `alibi_detect.utils.state.state`
## Constants
### `logger`
```python
logger: logging.Logger = <Logger alibi_detect.utils.state.state (WARNING)>
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

## `StateMixin`

_Inherits from:_ `ABC`

Utility class that provides methods to save and load stateful attributes to disk.

### Methods

#### `load_state`

```python
load_state(filepath: Union[str, os.PathLike])
```

Load the detector's state from disk, in order to restart from a checkpoint previously generated with

`save_state`.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `filepath` | `Union[str, os.PathLike]` |  | The directory to load state from. |

#### `save_state`

```python
save_state(filepath: Union[str, os.PathLike])
```

Save a detector's state to disk in order to generate a checkpoint.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `filepath` | `Union[str, os.PathLike]` |  | The directory to save state to. |
