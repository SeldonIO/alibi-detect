# `alibi_detect.saving.registry`

This registry allows Python objects to be registered and accessed by their string reference later on. The primary usage
is to register objects so that they can be specified in a `config.toml` file. A number of Alibi Detect functions are
also pre-registered in the registry for convenience. See the
`Registering artefacts <https://docs.seldon.io/projects/alibi-detect/en/stable/overview/config_files.html#registering-artefacts>`_  # noqa: E501
documentation.

## Constants
### `has_pytorch`
```python
has_pytorch: bool = True
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

### `has_keops`
```python
has_keops: bool = True
```
bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.
