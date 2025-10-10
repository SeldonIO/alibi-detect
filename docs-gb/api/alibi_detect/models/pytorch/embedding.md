# `alibi_detect.models.pytorch.embedding`
## `TransformerEmbedding`

_Inherits from:_ `Module`

### Constructor

```python
TransformerEmbedding(self, model_name_or_path: str, embedding_type: str, layers: List[int] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `model_name_or_path` | `str` |  |  |
| `embedding_type` | `str` |  |  |
| `layers` | `Optional[List[int]]` | `None` |  |

### Methods

#### `forward`

```python
forward(tokens: Dict[str, torch.Tensor]) -> torch.Tensor
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `tokens` | `Dict[str, torch.Tensor]` |  |  |

**Returns**
- Type: `torch.Tensor`

## Functions
### `hidden_state_embedding`

```python
hidden_state_embedding(hidden_states: torch.Tensor, layers: List[int], use_cls: bool, reduce_mean: bool = True) -> torch.Tensor
```

Extract embeddings from hidden attention state layers.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `hidden_states` | `torch.Tensor` |  | Attention hidden states in the transformer model. |
| `layers` | `List[int]` |  | List of layers to use for the embedding. |
| `use_cls` | `bool` |  | Whether to use the next sentence token (CLS) to extract the embeddings. |
| `reduce_mean` | `bool` | `True` | Whether to take the mean of the output tensor. |

**Returns**
- Type: `torch.Tensor`
