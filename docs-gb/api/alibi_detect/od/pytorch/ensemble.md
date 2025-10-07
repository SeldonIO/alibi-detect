# `alibi_detect.od.pytorch.ensemble`
## `AverageAggregator`

_Inherits from:_ `BaseTransformTorch`, `Module`

### Constructor

```python
AverageAggregator(self, weights: Optional[torch.Tensor] = None)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `weights` | `Optional[torch.Tensor]` | `None` | Optional parameter to weight the scores. If `weights` is left ``None`` then will be set to a vector of ones. |

### Methods

#### `transform`

```python
transform(scores: torch.Tensor) -> torch.Tensor
```

Averages the scores of the detectors in an ensemble. If weights were passed in the `__init__`

then these are used to weight the scores.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `scores` | `torch.Tensor` |  | `Torch.Tensor` of scores from ensemble of detectors. |

**Returns**
- Type: `torch.Tensor`

## `BaseTransformTorch`

_Inherits from:_ `Module`

### Constructor

```python
BaseTransformTorch(self)
```
### Methods

#### `forward`

```python
forward(x: torch.Tensor) -> torch.Tensor
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `torch.Tensor` |  |  |

**Returns**
- Type: `torch.Tensor`

#### `transform`

```python
transform(x: torch.Tensor)
```

Public transform method.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `torch.Tensor` |  | `torch.Tensor` array to be transformed |

## `Ensembler`

_Inherits from:_ `BaseTransformTorch`, `Module`, `FitMixinTorch`, `ABC`

### Constructor

```python
Ensembler(self, normalizer: Optional[alibi_detect.od.pytorch.ensemble.BaseTransformTorch] = None, aggregator: alibi_detect.od.pytorch.ensemble.BaseTransformTorch = None)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `normalizer` | `Optional[alibi_detect.od.pytorch.ensemble.BaseTransformTorch]` | `None` | `BaseFittedTransformTorch` object to normalize the scores. If ``None`` then no normalization is applied. |
| `aggregator` | `Optional[alibi_detect.od.pytorch.ensemble.BaseTransformTorch]` | `None` | `BaseTransformTorch` object to aggregate the scores. If ``None`` defaults to `AverageAggregator`. |

### Methods

#### `fit`

```python
fit(x: torch.Tensor) -> typing_extensions.Self
```

Fit the normalizer to the scores.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `torch.Tensor` |  | `Torch.Tensor` of scores from ensemble of detectors. |

**Returns**
- Type: `typing_extensions.Self`

#### `transform`

```python
transform(x: torch.Tensor) -> torch.Tensor
```

Apply the normalizer and aggregator to the scores.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `torch.Tensor` |  | `Torch.Tensor` of scores from ensemble of detectors. |

**Returns**
- Type: `torch.Tensor`

## `FitMixinTorch`

_Inherits from:_ `ABC`

### Methods

#### `check_fitted`

```python
check_fitted()
```

Checks to make sure object has been fitted.

#### `fit`

```python
fit(x_ref: torch.Tensor) -> typing_extensions.Self
```

Abstract fit method.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `torch.Tensor` |  |  |
| `x` |  |  | `torch.Tensor` to fit object on. |

**Returns**
- Type: `typing_extensions.Self`

## `MaxAggregator`

_Inherits from:_ `BaseTransformTorch`, `Module`

### Constructor

```python
MaxAggregator(self)
```
### Methods

#### `transform`

```python
transform(scores: torch.Tensor) -> torch.Tensor
```

Takes the maximum score of a set of detectors in an ensemble.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `scores` | `torch.Tensor` |  | `Torch.Tensor` of scores from ensemble of detectors. |

**Returns**
- Type: `torch.Tensor`

## `MinAggregator`

_Inherits from:_ `BaseTransformTorch`, `Module`

### Constructor

```python
MinAggregator(self)
```
### Methods

#### `transform`

```python
transform(scores: torch.Tensor) -> torch.Tensor
```

Takes the minimum score of a set of detectors in an ensemble.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `scores` | `torch.Tensor` |  | `Torch.Tensor` of scores from ensemble of detectors. |

**Returns**
- Type: `torch.Tensor`

## `PValNormalizer`

_Inherits from:_ `BaseTransformTorch`, `Module`, `FitMixinTorch`, `ABC`

### Constructor

```python
PValNormalizer(self)
```
### Methods

#### `fit`

```python
fit(val_scores: torch.Tensor) -> typing_extensions.Self
```

Fit transform on scores.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `val_scores` | `torch.Tensor` |  | score outputs of ensemble of detectors applied to reference data. |

**Returns**
- Type: `typing_extensions.Self`

#### `transform`

```python
transform(scores: torch.Tensor) -> torch.Tensor
```

Transform scores to 1 - p-values.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `scores` | `torch.Tensor` |  | `Torch.Tensor` of scores from ensemble of detectors. |

**Returns**
- Type: `torch.Tensor`

## `ShiftAndScaleNormalizer`

_Inherits from:_ `BaseTransformTorch`, `Module`, `FitMixinTorch`, `ABC`

### Constructor

```python
ShiftAndScaleNormalizer(self)
```
### Methods

#### `fit`

```python
fit(val_scores: torch.Tensor) -> typing_extensions.Self
```

Computes the mean and standard deviation of the scores and stores them.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `val_scores` | `torch.Tensor` |  | `Torch.Tensor` of scores from ensemble of detectors. |

**Returns**
- Type: `typing_extensions.Self`

#### `transform`

```python
transform(scores: torch.Tensor) -> torch.Tensor
```

Transform scores to normalized values. Subtracts the mean and scales by the standard deviation.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `scores` | `torch.Tensor` |  | `Torch.Tensor` of scores from ensemble of detectors. |

**Returns**
- Type: `torch.Tensor`

## `TopKAggregator`

_Inherits from:_ `BaseTransformTorch`, `Module`

### Constructor

```python
TopKAggregator(self, k: Optional[int] = None)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `k` | `Optional[int]` | `None` | number of scores to take the mean of. If `k` is left ``None`` then will be set to half the number of scores passed in the forward call. |

### Methods

#### `transform`

```python
transform(scores: torch.Tensor) -> torch.Tensor
```

Takes the mean of the top `k` scores.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `scores` | `torch.Tensor` |  | `Torch.Tensor` of scores from ensemble of detectors. |

**Returns**
- Type: `torch.Tensor`
