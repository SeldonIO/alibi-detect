# `alibi_detect.utils.saving.saving`
## Functions
### `load_detector`

```python
load_detector(filepath: Union[str, os.PathLike], kwargs) -> Union[alibi_detect.base.Detector, alibi_detect.base.ConfigurableDetector]
```

Load outlier, drift or adversarial detector.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `filepath` | `Union[str, os.PathLike]` |  | Load directory. |

**Returns**
- Type: `Union[alibi_detect.base.Detector, alibi_detect.base.ConfigurableDetector]`

### `save_detector`

```python
save_detector(detector: Union[alibi_detect.base.Detector, alibi_detect.base.ConfigurableDetector], filepath: Union[str, os.PathLike], legacy: bool = False) -> None
```

Save outlier, drift or adversarial detector.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `detector` | `Union[alibi_detect.base.Detector, alibi_detect.base.ConfigurableDetector]` |  | Detector object. |
| `filepath` | `Union[str, os.PathLike]` |  | Save directory. |
| `legacy` | `bool` | `False` | Whether to save in the legacy .dill format instead of via a config.toml file. Default is `False`. |

**Returns**
- Type: `None`
