# `alibi_detect.utils.frameworks`
## Constants
### `ERROR_TYPES`
```python
ERROR_TYPES: dict = {'prophet': 'prophet', 'tensorflow_probability': 'tensorflow', 'tensorflow': ...
```

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

### `has_keops`
```python
has_keops: bool = True
```
bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### `HAS_BACKEND`
```python
HAS_BACKEND: dict = {'tensorflow': True, 'pytorch': True, 'sklearn': True, 'keops': True}
```

## `BackendValidator`

### Constructor

```python
BackendValidator(self, backend_options: Dict[Optional[str], List[str]], construct_name: str)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `backend_options` | `Dict[Optional[str], List[str]]` |  | Dictionary from backend to list of dependencies that must be satisfied. The keys are the available options for the user and the values should be a list of dependencies that are checked via the `HAS_BACKEND` map defined in this module. An example of `backend_options` would be `{'tensorflow': ['tensorflow'], 'pytorch': ['pytorch'], None: []}`.This would mean `'tensorflow'`, `'pytorch'` or `None` are available backend options. If the user passes a different backend they will receive and error listing the correct backends. In addition, if one of the dependencies in the `backend_option` values is missing for the specified backend the validator will issue an error message telling the user what dependency bucket to install. |
| `construct_name` | `str` |  | Name of the object that has a set of backends we need to verify. |

### Methods

#### `verify_backend`

```python
verify_backend(backend: str)
```

Verifies backend choice.

Verifies backend is implemented and that the correct dependencies are installed for the requested backend. If
the backend is not implemented or a dependency is missing then an error is issued.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `backend` | `str` |  | Choice of backend the user wishes to initialize the alibi-detect construct with. Must be one of the keys in the `self.backend_options` dictionary. |

## `Framework`

_Inherits from:_ `str`, `Enum`

An enumeration.
