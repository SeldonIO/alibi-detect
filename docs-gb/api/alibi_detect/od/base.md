# `alibi_detect.od.base`
## `FittedTransformProtocol`

_Inherits from:_ `TransformProtocol`, `Protocol`, `Generic`

Protocol for fitted transformer objects.

This protocol models the joint interface of the :py:obj:`~alibi_detect.od.pytorch.ensemble.BaseTransformTorch`
class and the :py:obj:`~alibi_detect.od.pytorch.ensemble.FitMixinTorch` class. These objects are transforms that
require to be fit.

### Methods

#### `check_fitted`

```python
check_fitted()
```

#### `fit`

```python
fit(x_ref)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` |  |  |  |

#### `set_fitted`

```python
set_fitted()
```

## `TransformProtocol`

_Inherits from:_ `Protocol`, `Generic`

Protocol for transformer objects.

The :py:obj:`~alibi_detect.od.pytorch.ensemble.BaseTransformTorch` object provides abstract methods for
objects that map between `torch` tensors. This protocol models the interface of the `BaseTransformTorch`
class.

### Constructor

```python
TransformProtocol(self, *args, **kwargs)
```
### Methods

#### `transform`

```python
transform(x)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` |  |  |  |

## Functions
### `get_aggregator`

```python
get_aggregator(aggregator: Union[alibi_detect.od.base.TransformProtocol, Literal[TopKAggregator, AverageAggregator, MaxAggregator, MinAggregator]]) -> alibi_detect.od.base.TransformProtocol
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `aggregator` | `Union[alibi_detect.od.base.TransformProtocol, Literal[TopKAggregator, AverageAggregator, MaxAggregator, MinAggregator]]` |  |  |

**Returns**
- Type: `alibi_detect.od.base.TransformProtocol`

### `get_normalizer`

```python
get_normalizer(normalizer: Union[alibi_detect.od.base.TransformProtocol, alibi_detect.od.base.FittedTransformProtocol, Literal[PValNormalizer, ShiftAndScaleNormalizer]]) -> alibi_detect.od.base.TransformProtocol
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `normalizer` | `Union[alibi_detect.od.base.TransformProtocol, alibi_detect.od.base.FittedTransformProtocol, Literal[PValNormalizer, ShiftAndScaleNormalizer]]` |  |  |

**Returns**
- Type: `alibi_detect.od.base.TransformProtocol`
