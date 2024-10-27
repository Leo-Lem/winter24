from torch.nn import Module, Linear
from torch import Tensor, tanh
from torch.nn.functional import softmax


class Attention(Module):
    """ Calculate the attention weights (Bahdanau). """

    def __init__(self,
                 encoder_dim: int,
                 decoder_dim: int,
                 attention_dim: int):
        super(Attention, self).__init__()

        self.attention_dim = attention_dim
        self.DecoderToAttention = Linear(decoder_dim, attention_dim)
        self.EncoderToAttention = Linear(encoder_dim, attention_dim)
        self.AttentionScore = Linear(attention_dim, 1)

    def forward(self, features: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        # (batch_size,num_layers,attention_dim)
        encoded_features: Tensor = self.EncoderToAttention(features)
        # (batch_size,attention_dim)
        hidden_state_weight: Tensor = self.DecoderToAttention(hidden)

        # (batch_size,num_layers,attemtion_dim)
        combined_states = tanh(
            encoded_features + hidden_state_weight.unsqueeze(1))

        # (batch_size,num_layers,1)
        attention_scores: Tensor = self.AttentionScore(combined_states)
        # (batch_size,num_layers)
        attention_scores = attention_scores.squeeze(2)

        # (batch_size,num_layers)
        alpha = softmax(attention_scores, dim=1)

        # (batch_size,num_layers,features_dim)
        context = features * alpha.unsqueeze(2)
        # (batch_size,num_layers)
        context = context.sum(dim=1)

        return alpha, context
