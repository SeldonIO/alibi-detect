# `alibi_detect.saving.validate`
## Constants
### `DETECTOR_CONFIGS`
```python
DETECTOR_CONFIGS: dict = {'KSDrift': <class 'alibi_detect.saving.schemas.KSDriftConfig'>, 'ChiSquareDr...
```

### `DETECTOR_CONFIGS_RESOLVED`
```python
DETECTOR_CONFIGS_RESOLVED: dict = {'KSDrift': <class 'alibi_detect.saving.schemas.KSDriftConfigResolved'>, 'Chi...
```

## Functions
### `validate_config`

```python
validate_config(cfg: dict, resolved: bool = False) -> dict
```

Validates a detector config dict by passing the dict to the detector's pydantic model schema.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `cfg` | `dict` |  | The detector config dict. |
| `resolved` | `bool` | `False` | Whether the config is resolved or not. For example, if resolved=True, `x_ref` is expected to be a np.ndarray, wheras if resolved=False, `x_ref` is expected to be a str. |

**Returns**
- Type: `dict`
