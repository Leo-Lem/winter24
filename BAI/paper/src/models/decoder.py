from torch import Tensor, tensor, full
from torch.nn import Module, Embedding, GRUCell, Linear
from torch import float32

from data import Vocabulary


class GRUCaptionDecoder(Module):
    """A simplified GRU-based language model for generating image captions."""

    def __init__(self,
                 vocabulary_size: int,
                 embedding_size: int = 256,
                 hidden_size: int = 512):
        """
        Args:
            vocabulary_size: The number of tokens in the vocabulary.
            embedding_size: The size of the token embeddings.
            hidden_size: The size of the hidden state of the GRU.
        """
        super().__init__()
        self.embedding_layer = Embedding(vocabulary_size, embedding_size)
        self.feature_projection = Linear(2048, hidden_size)
        self.gru_cell = GRUCell(embedding_size, hidden_size)
        self.output_layer = Linear(hidden_size, vocabulary_size)

    def forward(self, features: Tensor, max_len: int) -> Tensor:
        """ Generate captions for the given image features.

        Args:
            features: Image features extracted from the encoder of shape (batch_size, num_features, feature_size).
            max_len: Maximum length of the generated captions.

        Returns:
            A tensor of shape (batch_size, max_len, vocabulary_size) containing logits for each token position.
        """
        assert features.dim() == 3, "Input features must have 3 dimensions."
        assert features.size(1) == 49, "Number of features must be 49."
        assert features.size(2) == 2048, "Feature size must be 2048."
        assert max_len > 0, "max_len must be a positive integer."

        batch_size = features.size(0)
        hidden = self.feature_projection(features.mean(dim=1))
        embeddings = self.embedding_layer(
            tensor(Vocabulary.sos_index, device=features.device).repeat(batch_size))

        logits: Tensor = full((batch_size, max_len, self.output_layer.out_features),
                              fill_value=Vocabulary.pad_index,
                              device=features.device,
                              dtype=float32)

        for _ in range(max_len):
            hidden = self.gru_cell(embeddings, hidden)
            output: Tensor = self.output_layer(hidden)

            logits[:, _, :] = output

            # Select predicted token IDs for next embedding
            predicted_ids = output.argmax(dim=1)
            embeddings = self.embedding_layer(predicted_ids)

            # Stop if all captions have generated an <EOS>
            if all(predicted_ids == Vocabulary.eos_index):
                break

        assert logits.dim() == 3, "Output logits must have 3 dimensions."
        assert logits.size(0) == batch_size, "Batch size mismatch."
        assert logits.size(1) == max_len, "Caption length mismatch."
        assert logits.size(2) == self.embedding_layer.num_embeddings, \
            "Vocabulary size mismatch."

        return logits
