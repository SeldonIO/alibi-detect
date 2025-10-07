# `alibi_detect.exceptions`

This module defines the Alibi Detect exception hierarchy and common exceptions used across the library.

## `AlibiDetectException`

_Inherits from:_ `Exception`, `BaseException`, `ABC`

### Constructor

```python
AlibiDetectException(self, message: str) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `message` | `str` |  | The error message. |

## `NotFittedError`

_Inherits from:_ `AlibiDetectException`, `Exception`, `BaseException`, `ABC`

### Constructor

```python
NotFittedError(self, object_name: str) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `object_name` | `str` |  | The name of the unfit object. |

## `ThresholdNotInferredError`

_Inherits from:_ `AlibiDetectException`, `Exception`, `BaseException`, `ABC`

### Constructor

```python
ThresholdNotInferredError(self, object_name: str) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `object_name` | `str` |  | The name of the object that does not have a threshold fit. |
