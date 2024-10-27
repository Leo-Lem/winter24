from torch import zeros, cat, Tensor, tensor
from torch.nn import Module, Embedding, LSTMCell, Linear, Dropout

from data import Vocabulary
from .attention import Attention


class CaptionDecoder(Module):
    """ An RNN-based language model that generates image captions. """

    def __init__(self,
                 vocab_size: int,
                 embed_size: int,
                 attention_dim: int,
                 encoder_dim: int,
                 decoder_dim: int,
                 drop_probability: float = 0.3):
        super(CaptionDecoder, self).__init__()

        self.vocab_size = vocab_size

        self.embedding_layer = Embedding(vocab_size, embed_size)
        self.attention_layer = Attention(
            encoder_dim, decoder_dim, attention_dim)
        self.init_cell = Linear(encoder_dim, decoder_dim)
        self.init_hidden = Linear(encoder_dim, decoder_dim)
        self.lstm_layer = LSTMCell(
            embed_size+encoder_dim, decoder_dim, bias=True)
        self.output_layer = Linear(decoder_dim, vocab_size)
        self.drop_layer = Dropout(drop_probability)

    def forward(self, features: Tensor, captions: Tensor) -> tuple[Tensor, Tensor]:
        # (batch_size, decoder_dim)
        hidden, cell = self._init_lstm_layer(features)

        length = len(captions[0])-1
        batch_size = captions.size(0)
        num_features = features.size(1)
        preds = zeros(batch_size, length, self.vocab_size)
        alphas = zeros(batch_size, length, num_features)

        embeddings = self.embedding_layer(captions)

        for i in range(length):
            alpha, context = self.attention_layer(features, hidden)
            lstm_input = cat((embeddings[:, i], context), dim=1)
            hidden, cell = self.lstm_layer(lstm_input, (hidden, cell))

            output = self.output_layer(self.drop_layer(hidden))

            preds[:, i] = output
            alphas[:, i] = alpha

        return preds, alphas

    def generate_caption(self, features: Tensor, vocab: Vocabulary, max_len: int = 20) -> tuple[str, list]:
        """ Generate captions for given image features using greedy search. """
        # (batch_size, decoder_dim)
        hidden, cell = self._init_lstm_layer(features)

        batch_size = features.size(0)

        word = tensor(vocab.token_to_id['<SOS>']).view(1, -1)
        embeddings = self.embedding_layer(word)

        alphas, captions = [], []
        for _ in range(max_len):
            alpha, context = self.attention_layer(features, hidden)

            alphas.append(alpha)

            lstm_input = cat((embeddings[:, 0], context), dim=1)
            hidden, cell = self.lstm_layer(lstm_input, (hidden, cell))
            output: Tensor = self.output_layer(
                self.drop_layer(hidden)).view(batch_size, -1)

            # Select the word with the highest score
            predicted_word = output.argmax(dim=1).item()

            # Stop appending words when <EOS> is predicted
            if vocab.id_to_token[predicted_word] == "<EOS>":
                break

            captions.append(predicted_word)

            # Use the predicted word as the next input
            embeddings = self.embedding_layer(
                tensor([predicted_word]).unsqueeze(0))

        # Convert indices to words and return the caption without <EOS>
        caption_text = [vocab.id_to_token[idx] for idx in captions]
        return caption_text, alphas

    def _init_lstm_layer(self, encoded: Tensor) -> tuple[Tensor, Tensor]:
        """ Initialize the hidden and cell states of the LSTM. """
        mean_encoded = encoded.mean(dim=1)
        return self.init_hidden(mean_encoded), self.init_cell(mean_encoded)
