# `alibi_detect.od.pytorch.base`
## `TorchOutlierDetector`

_Inherits from:_ `Module`, `FitMixinTorch`, `ABC`

Base class for torch backend outlier detection algorithms.

### Constructor

```python
TorchOutlierDetector(self, device: Union[typing_extensions.Literal['cuda', 'gpu', 'cpu'], ForwardRef('torch.device'), NoneType] = None)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `device` | `Union[Literal[cuda, gpu, cpu], torch.device, None]` | `None` |  |

### Methods

#### `check_threshold_inferred`

```python
check_threshold_inferred()
```

Check if threshold is inferred.

#### `infer_threshold`

```python
infer_threshold(x: torch.Tensor, fpr: float)
```

Infer the threshold for the data. Prerequisite for outlier predictions.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `torch.Tensor` |  | Data to infer the threshold for. |
| `fpr` | `float` |  | False positive rate to use for threshold inference. |

#### `predict`

```python
predict(x: torch.Tensor) -> alibi_detect.od.pytorch.base.TorchOutlierDetectorOutput
```

Predict outlier labels for the data.

Computes the outlier scores. If the detector is not fit on reference data we raise an error.
If the threshold is inferred, the outlier labels and p-values are also computed and returned.
Otherwise, the outlier labels and p-values are set to ``None``.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `torch.Tensor` |  | Data to predict. |

**Returns**
- Type: `alibi_detect.od.pytorch.base.TorchOutlierDetectorOutput`

#### `score`

```python
score(x: torch.Tensor) -> torch.Tensor
```

Score the data.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `torch.Tensor` |  | Data to score. |

**Returns**
- Type: `torch.Tensor`

## `TorchOutlierDetectorOutput`

Output of the outlier detector.

### Fields

| Field | Type | Default |
| ----- | ---- | ------- |
| `threshold_inferred` | `bool` | `` |
| `instance_score` | `torch.Tensor` | `` |
| `threshold` | `Optional[torch.Tensor]` | `` |
| `is_outlier` | `Optional[torch.Tensor]` | `` |
| `p_value` | `Optional[torch.Tensor]` | `` |

### Constructor

```python
TorchOutlierDetectorOutput(self, threshold_inferred: bool, instance_score: torch.Tensor, threshold: Optional[torch.Tensor], is_outlier: Optional[torch.Tensor], p_value: Optional[torch.Tensor]) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `threshold_inferred` | `bool` |  |  |
| `instance_score` | `torch.Tensor` |  |  |
| `threshold` | `Optional[torch.Tensor]` |  |  |
| `is_outlier` | `Optional[torch.Tensor]` |  |  |
| `p_value` | `Optional[torch.Tensor]` |  |  |

### Methods

#### `to_frontend_dtype`

```python
to_frontend_dtype()
```

## Functions
### `to_frontend_dtype`

```python
to_frontend_dtype(x: Union[torch.Tensor, alibi_detect.od.pytorch.base.TorchOutlierDetectorOutput]) -> Union[numpy.ndarray, Dict[str, numpy.ndarray]]
```

Converts any `torch` tensors found in input to `numpy` arrays.

Takes a `torch` tensor or `TorchOutlierDetectorOutput` and converts any `torch` tensors found to `numpy` arrays

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[torch.Tensor, alibi_detect.od.pytorch.base.TorchOutlierDetectorOutput]` |  | Data to convert. |

**Returns**
- Type: `Union[numpy.ndarray, Dict[str, numpy.ndarray]]`
