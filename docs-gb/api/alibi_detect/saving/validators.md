# `alibi_detect.saving.validators`
## Constants
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

### `T`
```python
T: TypeVar = ~T
```
Type variable.

Usage::

  T = TypeVar('T')  # Can be anything
  A = TypeVar('A', str, bytes)  # Must be str or bytes

Type variables exist primarily for the benefit of static type
checkers.  They serve as the parameters for generic types as well
as for generic function definitions.  See class Generic for more
information on generic types.  Generic functions work as follows:

  def repeat(x: T, n: int) -> List[T]:
      '''Return a list containing n references to x.'''
      return [x]*n

  def longest(x: A, y: A) -> A:
      '''Return the longest of two strings.'''
      return x if len(x) >= len(y) else y

The latter example's signature is essentially the overloading
of (str, str) -> str and (bytes, bytes) -> bytes.  Also note
that if the arguments are instances of some subclass of str,
the return type is still plain str.

At runtime, isinstance(x, T) and issubclass(C, T) will raise TypeError.

Type variables defined with covariant=True or contravariant=True
can be used to declare covariant or contravariant generic types.
See PEP 484 for more details. By default generic types are invariant
in all type variables.

Type variables can be introspected. e.g.:

  T.__name__ == 'T'
  T.__constraints__ == ()
  T.__covariant__ == False
  T.__contravariant__ = False
  A.__constraints__ == (str, bytes)

Note that only type variables defined in global scope can be pickled.

## `NDArray`

_Inherits from:_ `Generic`, `ndarray`

A Generic pydantic model to coerce to np.ndarray's.

### Methods

#### `validate`

```python
validate(val: typing.Any, field: pydantic.v1.fields.ModelField) -> Optional[numpy.ndarray]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `val` | `typing.Any` |  |  |
| `field` | `pydantic.v1.fields.ModelField` |  |  |

**Returns**
- Type: `Optional[numpy.ndarray]`

## Functions
### `coerce_2_tensor`

```python
coerce_2_tensor(value: Union[float, List[float]], values: dict)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `value` | `Union[float, List[float]]` |  |  |
| `values` | `dict` |  |  |

### `coerce_int2list`

```python
coerce_int2list(value: int) -> List[int]
```

Validator to coerce int to list (pydantic doesn't do this by default).

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `value` | `int` |  |  |

**Returns**
- Type: `List[int]`

### `validate_framework`

```python
validate_framework(framework: str, field: pydantic.v1.fields.ModelField) -> str
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `framework` | `str` |  |  |
| `field` | `pydantic.v1.fields.ModelField` |  |  |

**Returns**
- Type: `str`
