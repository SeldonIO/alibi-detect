# `alibi_detect.cd.utils`
## Constants
### `logger`
```python
logger: logging.Logger = <Logger alibi_detect.cd.utils (WARNING)>
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

## Functions
### `encompass_batching`

```python
encompass_batching(model: Callable, backend: str, batch_size: int, device: Union[Literal[cuda, gpu, cpu], ForwardRef('torch.device'), None] = None, preprocess_batch_fn: Optional[Callable] = None, tokenizer: Optional[Callable] = None, max_len: Optional[int] = None) -> Callable
```

Takes a function that must be batch evaluated (on tokenized input) and returns a function

that handles batching (and tokenization).

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `model` | `Callable` |  |  |
| `backend` | `str` |  |  |
| `batch_size` | `int` |  |  |
| `device` | `Union[Literal[cuda, gpu, cpu], ForwardRef('torch.device'), None]` | `None` |  |
| `preprocess_batch_fn` | `Optional[Callable]` | `None` |  |
| `tokenizer` | `Optional[Callable]` | `None` |  |
| `max_len` | `Optional[int]` | `None` |  |

**Returns**
- Type: `Callable`

### `encompass_shuffling_and_batch_filling`

```python
encompass_shuffling_and_batch_filling(model_fn: Callable, batch_size: int) -> Callable
```

Takes a function that already handles batching but additionally performing shuffling

and ensures instances are evaluated as part of full batches.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `model_fn` | `Callable` |  |  |
| `batch_size` | `int` |  |  |

**Returns**
- Type: `Callable`

### `get_input_shape`

```python
get_input_shape(shape: Optional[Tuple], x_ref: Union[numpy.ndarray, list]) -> Optional[Tuple]
```

Optionally infer shape from reference data.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `shape` | `Optional[Tuple]` |  |  |
| `x_ref` | `Union[numpy.ndarray, list]` |  |  |

**Returns**
- Type: `Optional[Tuple]`

### `update_reference`

```python
update_reference(X_ref: numpy.ndarray, X: numpy.ndarray, n: int, update_method: Optional[Dict[str, int]] = None) -> numpy.ndarray
```

Update reference dataset for drift detectors.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X_ref` | `numpy.ndarray` |  | Current reference dataset. |
| `X` | `numpy.ndarray` |  | New data. |
| `n` | `int` |  | Count of the total number of instances that have been used so far. |
| `update_method` | `Optional[Dict[str, int]]` | `None` | Dict with as key `reservoir_sampling` or `last` and as value n. `reservoir_sampling` will apply reservoir sampling with reservoir of size n while `last` will return (at most) the last n instances. |

**Returns**
- Type: `numpy.ndarray`
