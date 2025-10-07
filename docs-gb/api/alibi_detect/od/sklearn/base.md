# `alibi_detect.od.sklearn.base`
## `FitMixinSklearn`

_Inherits from:_ `ABC`

### Methods

#### `check_fitted`

```python
check_fitted()
```

Checks to make sure object has been fitted.

#### `fit`

```python
fit(x_ref: numpy.ndarray) -> typing_extensions.Self
```

Abstract fit method.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `numpy.ndarray` |  |  |
| `x` |  |  | `torch.Tensor` to fit object on. |

**Returns**
- Type: `typing_extensions.Self`

## `SklearnOutlierDetector`

_Inherits from:_ `FitMixinSklearn`, `ABC`

Base class for sklearn backend outlier detection algorithms.

### Methods

#### `check_threshold_inferred`

```python
check_threshold_inferred()
```

Check if threshold is inferred.

#### `infer_threshold`

```python
infer_threshold(x: numpy.ndarray, fpr: float) -> None
```

Infer the threshold for the data. Prerequisite for outlier predictions.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `numpy.ndarray` |  | Data to infer the threshold for. |
| `fpr` | `float` |  | False positive rate to use for threshold inference. |

**Returns**
- Type: `None`

#### `predict`

```python
predict(x: numpy.ndarray) -> alibi_detect.od.sklearn.base.SklearnOutlierDetectorOutput
```

Predict outlier labels for the data.

Computes the outlier scores. If the detector is not fit on reference data we raise an error.
If the threshold is inferred, the outlier labels and p-values are also computed and returned.
Otherwise, the outlier labels and p-values are set to ``None``.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `numpy.ndarray` |  | Data to predict. |

**Returns**
- Type: `alibi_detect.od.sklearn.base.SklearnOutlierDetectorOutput`

#### `score`

```python
score(x: numpy.ndarray) -> numpy.ndarray
```

Score the data.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `numpy.ndarray` |  | Data to score. |

**Returns**
- Type: `numpy.ndarray`

## `SklearnOutlierDetectorOutput`

Output of the outlier detector.

### Fields

| Field | Type | Default |
| ----- | ---- | ------- |
| `threshold_inferred` | `bool` | `` |
| `instance_score` | `numpy.ndarray` | `` |
| `threshold` | `Optional[numpy.ndarray]` | `` |
| `is_outlier` | `Optional[numpy.ndarray]` | `` |
| `p_value` | `Optional[numpy.ndarray]` | `` |

### Constructor

```python
SklearnOutlierDetectorOutput(self, threshold_inferred: bool, instance_score: numpy.ndarray, threshold: Optional[numpy.ndarray], is_outlier: Optional[numpy.ndarray], p_value: Optional[numpy.ndarray]) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `threshold_inferred` | `bool` |  |  |
| `instance_score` | `numpy.ndarray` |  |  |
| `threshold` | `Optional[numpy.ndarray]` |  |  |
| `is_outlier` | `Optional[numpy.ndarray]` |  |  |
| `p_value` | `Optional[numpy.ndarray]` |  |  |
