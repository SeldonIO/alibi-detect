from functools import partial
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, List


def hidden_state_embedding(hidden_states: torch.Tensor, layers: List[int],
                           use_cls: bool, reduce_mean: bool = True) -> torch.Tensor:
    """
    Extract embeddings from hidden attention state layers.

    Parameters
    ----------
    hidden_states
        Attention hidden states in the transformer model.
    layers
        List of layers to use for the embedding.
    use_cls
        Whether to use the next sentence token (CLS) to extract the embeddings.
    reduce_mean
        Whether to take the mean of the output tensor.

    Returns
    -------
    Tensor with embeddings.
    """
    hs = [hidden_states[layer][:, 0:1, :] if use_cls else hidden_states[layer] for layer in layers]
    hs = torch.cat(hs, dim=1)  # type: ignore
    y = hs.mean(dim=1) if reduce_mean else hs  # type: ignore
    return y


class TransformerEmbedding(nn.Module):
    def __init__(self, model_name_or_path: str, embedding_type: str, layers: List[int] = None) -> None:
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        self.emb_type = embedding_type
        self.hs_emb = partial(hidden_state_embedding, layers=layers, use_cls=embedding_type.endswith('cls'))

    def forward(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.model(**tokens)
        if self.emb_type == 'pooler_output':
            return output.pooler_output
        elif self.emb_type == 'last_hidden_state':
            return output.last_hidden_state.mean(dim=1)
        attention_hidden_states = output.hidden_states[1:]
        if self.emb_type.startswith('hidden_state'):
            return self.hs_emb(attention_hidden_states)
        else:
            raise ValueError('embedding_type needs to be one of pooler_output, '
                             'last_hidden_state, hidden_state, or hidden_state_cls.')
