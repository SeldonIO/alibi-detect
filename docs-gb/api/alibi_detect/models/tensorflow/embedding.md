# `alibi_detect.models.tensorflow.embedding`
## `TransformerEmbedding`

_Inherits from:_ `Model`, `TensorFlowTrainer`, `Trainer`, `Layer`, `TFLayer`, `KerasAutoTrackable`, `AutoTrackable`, `Trackable`, `Operation`, `KerasSaveable`

### Constructor

```python
TransformerEmbedding(self, model_name_or_path: str, embedding_type: str, layers: List[int] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `model_name_or_path` | `str` |  | Name of or path to the model. |
| `embedding_type` | `str` |  | Type of embedding to extract. Needs to be one of pooler_output, last_hidden_state, hidden_state or hidden_state_cls. |
| `layers` | `Optional[List[int]]` | `None` | If "hidden_state" or "hidden_state_cls" is used as embedding type, layers has to be a list with int's referring to the hidden layers used to extract the embedding. |
| `From` |  |  |  |
| `Last` |  |  | (classification token) further processed by a Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence prediction (classification) objective during pre-training. This output is usually not a good summary of the semantic content of the input, you’re often better with averaging or pooling the sequence of hidden-states for the whole input sequence. - last_hidden_state Sequence of hidden-states at the output of the last layer of the model. - hidden_state Hidden states of the model at the output of each layer. - hidden_state_cls See hidden_state but use the CLS token output. |

### Methods

#### `call`

```python
call(tokens: Dict[str, tensorflow.python.framework.tensor.Tensor]) -> tensorflow.python.framework.tensor.Tensor
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `tokens` | `Dict[str, tensorflow.python.framework.tensor.Tensor]` |  |  |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`

## Functions
### `hidden_state_embedding`

```python
hidden_state_embedding(hidden_states: tensorflow.python.framework.tensor.Tensor, layers: List[int], use_cls: bool, reduce_mean: bool = True) -> tensorflow.python.framework.tensor.Tensor
```

Extract embeddings from hidden attention state layers.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `hidden_states` | `tensorflow.python.framework.tensor.Tensor` |  | Attention hidden states in the transformer model. |
| `layers` | `List[int]` |  | List of layers to use for the embedding. |
| `use_cls` | `bool` |  | Whether to use the next sentence token (CLS) to extract the embeddings. |
| `reduce_mean` | `bool` | `True` | Whether to take the mean of the output tensor. |

**Returns**
- Type: `tensorflow.python.framework.tensor.Tensor`
