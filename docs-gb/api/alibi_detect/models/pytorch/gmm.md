# `alibi_detect.models.pytorch.gmm`
## `GMMModel`

_Inherits from:_ `Module`

### Constructor

```python
GMMModel(self, n_components: int, dim: int) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `n_components` | `int` |  | The number of mixture components. |
| `dim` | `int` |  | The dimensionality of the data. |

### Methods

#### `forward`

```python
forward(x: torch.Tensor) -> torch.Tensor
```

Compute the log-likelihood of the data.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `torch.Tensor` |  | Data to score. |

**Returns**
- Type: `torch.Tensor`
