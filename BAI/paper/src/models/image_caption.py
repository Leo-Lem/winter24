from torch import device, Tensor, save, no_grad, cuda
from torch.nn import Module, CrossEntropyLoss
from torchvision import transforms as T
from torch.optim import Adam
from matplotlib import pyplot as plt
from typing import Tuple

from ..data import FlickrDataset
from .encoder import ImageEncoder
from .decoder import CaptionDecoder


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
