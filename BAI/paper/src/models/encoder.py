from torch import Tensor
from torch.nn import Module, Sequential
from torchvision.models import resnet50, ResNet


class ResnetImageEncoder(Module):
    """ Image encoder using a pretrained ResNet-50 model. """

    def __init__(self):
        super().__init__()
        self.dimension = 2048
        self.model = self._get_passive_resnet()

    def forward(self, images: Tensor) -> Tensor:
        """ Extract features from a batch of image tensors.

        Args:
            images (Tensor): Batch of image tensors of shape (batch_size, channels=3, height=224, width=224).

        Returns:
            Tensor: Encoded image features of shape (batch_size, num_features=49, feature_size=2048).
        """
        assert images.dim() == 4, "Input images must be 4-dimensional."
        assert images.size(1) == 3, "Input images must have 3 channels."
        assert images.size(2) == 224, "Images must have size 224x224."

        features: Tensor = self.model(images)  # (batch_size, 2048, 7, 7)
        features = features.permute(0, 2, 3, 1)  # (batch_size, 7, 7, 2048)
        features = features.view(features.size(0), -1, features.size(-1))

        assert features.dim() == 3, "Output must have 3 dimensions."
        assert features.size(0) == images.size(0), \
            "Output batch size must match input batch size."
        assert features.size(1) == 49, "Output must have 49 features."
        assert features.size(2) == self.dimension, \
            "Output feature size must be 2048."

        return features

    def _get_passive_resnet(self) -> Sequential:
        """ Prepare the ResNet-50 model by setting it to evaluation mode, freezing parameters, and removing the final layers to retain spatial information in the output.

        Returns:
            Sequential: Modified ResNet-50 model as a Sequential module.
        """
        resnet: ResNet = resnet50(weights='ResNet50_Weights.DEFAULT').eval()
        for param in resnet.parameters():
            param.requires_grad_(False)
        return Sequential(*list(resnet.children())[:-2])
