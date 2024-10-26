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
