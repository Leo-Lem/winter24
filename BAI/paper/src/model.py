

import os
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pandas import read_csv
from torch import tensor, Tensor, cat
from torchvision.transforms import Compose




class FlickrDataset(Dataset):
    """ Flickr dataset for image captioning. """

    def __init__(self, path: str, captions: str = "captions.txt", transform: Compose = None, threshold: int = 5):
        self.path = path
        self.images_with_captions = read_csv(os.path.join(self.path, captions))
        self.transform = transform
        self.vocab = Vocabulary(threshold, self.captions)

    def __len__(self) -> int: return len(self.captions)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        raw_caption = self.captions[index]
        numericalized_caption = [self.vocab.token_to_id["<SOS>"]] + \
            self.vocab.numericalize(raw_caption) + \
            [self.vocab.token_to_id["<EOS>"]]
        caption = tensor(numericalized_caption)

        image_path = os.path.join(self.path, "Images", self.images[index])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, caption

    @property
    def images(self) -> list[str]:
        return self.images_with_captions["image"].tolist()

    @property
    def captions(self) -> list[str]:
        return self.images_with_captions["caption"].tolist()

    def data_loader(self, batch_size: int = 4, num_workers: int = 1, shuffle: bool = True):
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=FlickrDataset.CapsCollate(
                pad_index=self.vocab.token_to_id["<PAD>"], batch_first=True)
        )

    class CapsCollate:
        """ Collate to apply the padding to the captions with dataloader. """

        def __init__(self, pad_index: int, batch_first: bool = False):
            self.pad_index = pad_index
            self.batch_first = batch_first

        def __call__(self, batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
            """ Apply padding to the captions. """
            images = [item[0].unsqueeze(0) for item in batch]
            images = cat(images, dim=0)

            targets = [item[1] for item in batch]
            targets = pad_sequence(
                targets, batch_first=self.batch_first, padding_value=self.pad_index)

            return images, targets
from collections import Counter
from importlib import import_module
from spacy import Language
from spacy.cli import download


class Vocabulary:
    """ Vocabulary object to store the mapping between words and indices. """

    def __init__(self, threshold: int, sentences: list[str] = [], spacy_model_name: str = "en_core_web_sm"):
        self.threshold = threshold
        self.id_to_token = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.spacy_model = self._download_and_init_nlp(spacy_model_name)

        self._learn(sentences)

    def __len__(self) -> int: return len(self.id_to_token)

    @property
    def token_to_id(self) -> dict[str, int]:
        return {v: k for k, v in self.id_to_token.items()}

    def numericalize(self, text: str) -> list[int]:
        """ For each token in the text, return the index in the vocabulary. """
        token_to_id = self.token_to_id
        return [token_to_id[token] if token in token_to_id else token_to_id["<UNK>"]
                for token in self._tokenize(text)]

    def _learn(self, sentences: list[str]):
        """ Build the vocabulary from the sentences. """
        freq = Counter()
        for sentence in sentences:
            for token in self._tokenize(sentence):
                freq[token] += 1

                if token not in self.id_to_token.values():
                    self.id_to_token[len(self.id_to_token)] = token

    def _tokenize(self, text: str) -> list[str]:
        """ Tokenize the text using spacy model. """
        return [token.text.lower()
                for token in self.spacy_model.tokenizer(text)]

    def _download_and_init_nlp(self, model_name: str) -> Language:
        """Load a spaCy model, download it if it has not been installed yet.
        :param model_name: the model name, e.g., en_core_web_sm
        :param kwargs: options passed to the spaCy loader, such as component exclusion
        :return: an initialized spaCy Language
        """
        try:
            model_module = import_module(model_name)
        except ModuleNotFoundError:
            download(model_name)
            model_module = import_module(model_name)

        return model_module.load()

from torch.nn import Module, Linear
from torch import Tensor, tanh
from typing import Tuple
from torch.nn.functional import softmax


class Attention(Module):
    """ Calculate the attention weights (Bahdanau). """

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        super(Attention, self).__init__()

        self.attention_dim = attention_dim
        self.DecoderToAttention = Linear(decoder_dim, attention_dim)
        self.EncoderToAttention = Linear(encoder_dim, attention_dim)
        self.AttentionScore = Linear(attention_dim, 1)

    def forward(self, features: Tensor, hidden: Tensor) -> Tuple[Tensor, Tensor]:
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
from torch import zeros, device, cuda, cat, Tensor, tensor
from torch.nn import Module, Embedding, LSTMCell, Linear, Dropout
from typing import Tuple





class CaptionDecoder(Module):
    """ CaptionDecoder is a RNN-based language model that generates image captions. """

    def __init__(self,
                 embed_size: int,
                 vocab_size: int,
                 attention_dim: int,
                 encoder_dim: int,
                 decoder_dim: int,
                 drop_prob: float = 0.3,
                 device: device = device("cuda" if cuda.is_available() else "cpu")):
        super(CaptionDecoder, self).__init__()

        self.device = device
        self.vocab_size = vocab_size

        self.embedding_layer = Embedding(vocab_size, embed_size)
        self.attention_layer = Attention(
            encoder_dim, decoder_dim, attention_dim)
        self.init_cell = Linear(encoder_dim, decoder_dim)
        self.init_hidden = Linear(encoder_dim, decoder_dim)
        self.lstm_layer = LSTMCell(
            embed_size+encoder_dim, decoder_dim, bias=True)
        self.output_layer = Linear(decoder_dim, vocab_size)
        self.drop_layer = Dropout(drop_prob)

    def forward(self, features: Tensor, captions: Tensor) -> Tuple[Tensor, Tensor]:
        # (batch_size, decoder_dim)
        hidden, cell = self._init_lstm_layer(features)

        length = len(captions[0])-1
        batch_size = captions.size(0)
        num_features = features.size(1)
        preds = zeros(batch_size, length, self.vocab_size).to(self.device)
        alphas = zeros(batch_size, length, num_features).to(self.device)

        embeddings = self.embedding_layer(captions)

        for i in range(length):
            alpha, context = self.attention_layer(features, hidden)
            lstm_input = cat((embeddings[:, i], context), dim=1)
            hidden, cell = self.lstm_layer(lstm_input, (hidden, cell))

            output = self.output_layer(self.drop_layer(hidden))

            preds[:, i] = output
            alphas[:, i] = alpha

        return preds, alphas

    def generate_caption(self, features: Tensor, max_len: int = 20, vocab: Vocabulary = None) -> Tuple[str, list]:
        """ Generate captions for given image features using greedy search. """
        # (batch_size, decoder_dim)
        hidden, cell = self._init_lstm_layer(features)

        batch_size = features.size(0)

        word = tensor(vocab.token_to_id['<SOS>']).view(1, -1).to(self.device)
        embeddings = self.embedding_layer(word)

        alphas, captions = [], []
        for _ in range(max_len):
            alpha, context = self.attention_layer(features, hidden)

            alphas.append(self._get_apla_score(alpha))

            lstm_input = cat((embeddings[:, 0], context), dim=1)
            hidden, cell = self.lstm_layer(lstm_input, (hidden, cell))
            output: Tensor = self.output_layer(
                self.drop_layer(hidden)).view(batch_size, -1)

            # select the word with most val
            predicted_word = output.argmax(dim=1)

            # save the generated word
            captions.append(predicted_word.item())

            if vocab.id_to_token[predicted_word.item()] == "<EOS>":
                break

            embeddings = self.embedding_layer(predicted_word.unsqueeze(0))

        return [vocab.id_to_token[caption] for caption in captions], alphas

    def _get_apla_score(self, alpha: Tensor) -> Tensor:
        """ Get the attention score. """
        return alpha.cpu().detach().numpy()

    def _init_lstm_layer(self, encoded: Tensor) -> Tuple[Tensor, Tensor]:
        """ Initialize the hidden and cell states of the LSTM. """
        mean_encoded = encoded.mean(dim=1)
        return self.init_hidden(mean_encoded), self.init_cell(mean_encoded)
from torch import Tensor
from torch.nn import Module, Sequential
from torchvision.models import resnet50, ResNet


class ImageEncoder(Module):
    """ A CNN-based ResNet-50 model that is used to encode the images to a feature vector. """

    def __init__(self):
        """ Load the pretrained ResNet-50 and replace top fc layer. """
        super(ImageEncoder, self).__init__()

        resnet: ResNet = resnet50(weights='ResNet50_Weights.DEFAULT')

        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-2]
        self.model = Sequential(*modules)

    def forward(self, images: Tensor) -> Tensor:
        """ Extract feature vectors from input images. """
        # (batch_size,2048,7,7)
        features: Tensor = self.model(images)

        # (batch_size,7,7,2048)
        features = features.permute(0, 2, 3, 1)

        # (batch_size,49,2048)
        features = features.view(
            features.size(0), -1, features.size(-1))

        return features
from torch import device, Tensor, save, no_grad, cuda
from torch.nn import Module, CrossEntropyLoss
from torchvision import transforms as T
from torch.optim import Adam
from matplotlib import pyplot as plt
from typing import Tuple






class ImageCaption(Module):
    def __init__(self,
                 path: str,
                 batch_size: int,
                 num_workers: int,
                 embed_size: int,
                 attention_dim: int,
                 encoder_dim: int,
                 decoder_dim: int,
                 learning_rate: float,
                 transform: T.Compose = None,
                 device=device("cuda" if cuda.is_available() else "cpu")):
        super(ImageCaption, self).__init__()

        self.device = device

        self.embed_size = embed_size
        self.attention_dim = attention_dim
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

        self.dataset = FlickrDataset(
            path=path,
            transform=transform
        )
        self.loader = self.dataset.data_loader(
            batch_size=batch_size, num_workers=num_workers)
        vocab_size = len(self.dataset.vocab)

        self.encoder = ImageEncoder()
        self.decoder = CaptionDecoder(
            embed_size=embed_size,
            vocab_size=vocab_size,
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            device=device
        ).to(device)

        self.criterion = CrossEntropyLoss(
            ignore_index=self.dataset.vocab.token_to_id["<PAD>"])
        self.optimizer = Adam(self.parameters(), lr=learning_rate)

    def forward(self, images: Tensor, captions: Tensor) -> Tensor:
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def train(self, epochs: int = 5, print_every: int = 10):
        for epoch in range(epochs):
            for idx, (images, captions) in enumerate(iter(self.loader)):
                images, captions = images.to(
                    self.device), captions.to(self.device)

                # Zero the gradients.
                self.optimizer.zero_grad()

                # Feed forward
                outputs, _ = self(images, captions)

                # Calculate the batch loss.
                targets = captions[:, 1:]
                loss: CrossEntropyLoss = self.criterion(
                    outputs.view(-1, len(self.dataset.vocab)), targets.reshape(-1))

                # Backward pass.
                loss.backward()

                # Update the parameters in the optimizer.
                self.optimizer.step()

                if (idx+1) % print_every == 0:
                    print("Epoch: {} loss: {:.5f}".format(epoch+1, loss.item()))

                # generate the caption
                self.eval()
                with no_grad():
                    dataiter = iter(self.loader)
                    images, _ = next(dataiter)
                    features = self.encoder(images[0:1].to(self.device))
                    tokens, _ = self.decoder.generate_caption(
                        features, vocab=self.dataset.vocab)
                    self._show_image(images[0], title=str.join(' ', tokens))

                self.train()

            self._save(epoch+1)

    def predict(self, images: Tensor) -> Tuple[list, Tensor]:
        self.eval()

        with no_grad():
            features = self.encoder(images.to(self.device))
            tokens, alphas = self.decoder.generate_caption(
                features, vocab=self.dataset.vocab)
            self._show_image(images[0], title=str.join(' ', tokens))

        return tokens, alphas

    # def plot_attention(self, image: Tensor, result: list, attention_plot: Tensor):
    #     fig = plt.figure(figsize=(15, 15))

    #     for l in range(len(result)):
    #         temp_att = attention_plot[l].reshape(7, 7)

    #         subplot = fig.add_subplot(len(result)//2, len(result)//2, l+1)
    #         subplot.set_title(result[l])
    #         image = subplot.imshow(self._unnormalize(image))
    #         subplot.imshow(temp_att, cmap='gray', alpha=0.7,
    #                        extent=image.get_extent())

    #     plt.tight_layout()
    #     plt.show()

    def _save(self, epoch: int):
        save({
            'num_epochs': epoch,
            'embed_size': self.embed_size,
            'vocab_size': self.decoder.vocab_size,
            'attention_dim': self.attention_dim,
            'encoder_dim': self.encoder_dim,
            'decoder_dim': self.decoder_dim,
            'state_dict': self.state_dict()
        }, 'imagecaption_model_state.pth')

    def _unnormalize(self, image: Tensor):
        image[0] = image[0] * 0.229
        image[1] = image[1] * 0.224
        image[2] = image[2] * 0.225
        image[0] += 0.485
        image[1] += 0.456
        image[2] += 0.406

        return image.numpy().transpose((1, 2, 0))

    def _show_image(self, image: Tensor, title: str):
        """ Show image with caption and pause so that plots are updated. """
        plt.imshow(self._unnormalize(image))
        if title is not None:
            plt.title(title)
        plt.pause(.001)
from torch import cuda
from torchvision import transforms as T



model = ImageCaption(
    "flickr8k",
    batch_size=4,
    num_workers=2,
    embed_size=300,
    attention_dim=256,
    encoder_dim=2048,
    decoder_dim=512,
    learning_rate=3e-4,
    transform=T.Compose([
        T.Resize(226),
        T.RandomCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),
    device="cuda" if cuda.is_available() else "cpu"
)

model.train(epochs=3)

print(model.predict(model.dataset[0][0]))
