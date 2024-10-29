from torch import Tensor, tensor
from torch.nn import Module, Embedding, GRUCell, Linear
from data import Vocabulary


class SimpleCaptionDecoder(Module):
    """A simplified GRU-based language model for generating image captions."""

    def __init__(self, vocab: Vocabulary, embed_size: int = 256, decoder_dim: int = 512):
        """
        Args:
            vocab (Vocabulary): Vocabulary instance for token-to-id mappings.
            encoder_dim (int): Dimension of the encoder output features.
            embed_size (int): Size of the word embedding vectors (default: 256).
            decoder_dim (int): Dimension of the GRU hidden state (default: 512).
        """
        super().__init__()

        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embed_size = embed_size
        self.decoder_dim = decoder_dim

        # Layers for embedding, GRU, and output projection
        self.embedding_layer = Embedding(self.vocab_size, self.embed_size)
        self.gru_cell = GRUCell(self.embed_size, self.decoder_dim)
        self.output_layer = Linear(self.decoder_dim, self.vocab_size)

    def forward(self, features: Tensor, max_len: int = 20) -> list[str]:
        """
        Generate captions for a batch of image features using greedy decoding.

        Args:
            features (Tensor): Encoded image features of shape (batch_size, encoder_dim).
            max_len (int): Maximum length for generated captions (default: 20).

        Returns:
            list[str]: List of generated captions for each image in the batch.
        """
        batch_size = features.size(0)

        # Initialize the hidden state by averaging features across spatial dimensions
        hidden = features.mean(dim=1)  # (batch_size, decoder_dim)

        # Initialize word embeddings with the <SOS> token for each image in the batch
        start_token = tensor(
            self.vocab.token_to_id['<SOS>']).repeat(batch_size)
        embeddings = self.embedding_layer(
            start_token)  # (batch_size, embed_size)

        captions = [[] for _ in range(batch_size)]
        complete = [False] * batch_size  # Track completion of each caption

        for _ in range(max_len):
            # Forward pass through GRU cell
            # (batch_size, decoder_dim)
            hidden = self.gru_cell(embeddings, hidden)

            # Compute output logits for vocabulary prediction
            output = self.output_layer(hidden)  # (batch_size, vocab_size)
            # Select word with highest probability for each sample
            predicted_ids = output.argmax(dim=1)

            # Append predicted words to captions, and check for <EOS> to mark caption completion
            for i in range(batch_size):
                if not complete[i]:  # Only add if caption isn't complete
                    word_id = predicted_ids[i].item()
                    if self.vocab.id_to_token[word_id] == "<EOS>":
                        complete[i] = True
                    else:
                        captions[i].append(word_id)

            # If all captions are complete, break out of the loop
            if all(complete):
                break

            # Prepare embeddings for the next time step
            embeddings = self.embedding_layer(predicted_ids)

        # Convert word IDs to tokens and join them into sentences
        caption_texts = [" ".join([self.vocab.id_to_token[idx]
                                  for idx in caption]) for caption in captions]
        return caption_texts
