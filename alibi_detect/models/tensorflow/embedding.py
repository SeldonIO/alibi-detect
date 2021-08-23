from functools import partial
import tensorflow as tf
from transformers import TFAutoModel, AutoConfig
from typing import Dict, List


def hidden_state_embedding(hidden_states: tf.Tensor, layers: List[int],
                           use_cls: bool, reduce_mean: bool = True) -> tf.Tensor:
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
    hs = tf.concat(hs, axis=1)
    y = tf.reduce_mean(hs, axis=1) if reduce_mean else hs
    return y


class TransformerEmbedding(tf.keras.Model):
    def __init__(
            self,
            model_name_or_path: str,
            embedding_type: str,
            layers: List[int] = None
    ) -> None:
        """
        Extract text embeddings from transformer models.

        Parameters
        ----------
        model_name_or_path
            Name of or path to the model.
        embedding_type
            Type of embedding to extract. Needs to be one of pooler_output,
            last_hidden_state, hidden_state or hidden_state_cls.

            From the HuggingFace documentation:

            - pooler_output
                Last layer hidden-state of the first token of the sequence
                (classification token) further processed by a Linear layer and a Tanh
                activation function. The Linear layer weights are trained from the next
                sentence prediction (classification) objective during pre-training.
                This output is usually not a good summary of the semantic content of the
                input, you’re often better with averaging or pooling the sequence of
                hidden-states for the whole input sequence.
            - last_hidden_state
                Sequence of hidden-states at the output of the last layer of the model.
            - hidden_state
                Hidden states of the model at the output of each layer.
            - hidden_state_cls
                See hidden_state but use the CLS token output.
        layers
            If "hidden_state" or "hidden_state_cls" is used as embedding
            type, layers has to be a list with int's referring to the hidden layers used
            to extract the embedding.
        """
        super(TransformerEmbedding, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path, output_hidden_states=True)
        self.model = TFAutoModel.from_pretrained(model_name_or_path, config=self.config)
        self.emb_type = embedding_type
        self.hs_emb = partial(hidden_state_embedding, layers=layers, use_cls=embedding_type.endswith('cls'))

    def call(self, tokens: Dict[str, tf.Tensor]) -> tf.Tensor:
        output = self.model(tokens)
        if self.emb_type == 'pooler_output':
            return output.pooler_output
        elif self.emb_type == 'last_hidden_state':
            return tf.reduce_mean(output.last_hidden_state, axis=1)
        attention_hidden_states = output.hidden_states[1:]
        if self.emb_type.startswith('hidden_state'):
            return self.hs_emb(attention_hidden_states)
        else:
            raise ValueError('embedding_type needs to be one of pooler_output, '
                             'last_hidden_state, hidden_state, or hidden_state_cls.')
