from torch import Tensor
from PIL.Image import Image


from .checkpoint import CheckpointModule
from .encoder import ResnetImageEncoder
from .decoder import SimpleCaptionDecoder


class ImageCaption(CheckpointModule):
    def __init__(self, encoder: ResnetImageEncoder, decoder: SimpleCaptionDecoder):
        super().__init__("imagecaption.checkpoint")
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images: list[Image]) -> list[str]:
        """
        Generate captions for a batch of images.

        Args:
            images (list[Image]): A batch of PIL images to be captioned.

        Returns:
            list[str]: Generated captions for each image in the batch.
        """
        features = self.encoder(images)  # (batch_size, 49, encoder_dim)
        captions = self.decoder(features)  # (batch_size, max_len)
        return captions
