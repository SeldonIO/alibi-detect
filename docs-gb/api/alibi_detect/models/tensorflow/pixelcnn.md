# `alibi_detect.models.tensorflow.pixelcnn`
## Constants
### `absolute_import`
```python
absolute_import: __future__._Feature = _Feature((2, 5, 0, 'alpha', 1), (3, 0, 0, 'alpha', 0), 262144)
```

### `division`
```python
division: __future__._Feature = _Feature((2, 2, 0, 'alpha', 2), (3, 0, 0, 'alpha', 0), 131072)
```

### `print_function`
```python
print_function: __future__._Feature = _Feature((2, 6, 0, 'alpha', 2), (3, 0, 0, 'alpha', 0), 1048576)
```

## `Shift`

_Inherits from:_ `Bijector`, `Module`, `AutoTrackable`, `Trackable`

### Constructor

```python
Shift(self, shift, validate_args=False, name='shift')
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `shift` |  |  | Floating-point `Tensor`. |
| `validate_args` |  | `False` | Python `bool` indicating whether arguments should be checked for correctness. |
| `name` |  | `'shift'` | Python `str` name given to ops managed by this object. |

### Properties

| Property | Type | Description |
| -------- | ---- | ----------- |
| `shift` | `` | The `shift` `Tensor` in `Y = X + shift`. |
