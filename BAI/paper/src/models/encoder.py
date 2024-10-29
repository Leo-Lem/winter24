from torch import Tensor, stack
from torch.nn import Module, Sequential
from torchvision.models import resnet50, ResNet
from PIL.Image import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


class ResnetImageEncoder(Module):
    """ Image encoder using a pretrained ResNet-50 model. """

    def __init__(self):
        super().__init__()
        self.dimension = 2048
        self.model = self._get_passive_resnet()

    def forward(self, images: list[Image]) -> Tensor:
        """ Extract features from a batch of images.

        Args:
            images (list[Image]): List of PIL images.

        Returns:
            Tensor: Encoded features with shape (batch_size, 49, 2048).
        """
        images = self.preprocess_images(images)  # (batch_size, 3, 224, 224)
        features: Tensor = self.model(images)  # (batch_size, 2048, 7, 7)
        features = features.permute(0, 2, 3, 1)  # (batch_size, 7, 7, 2048)
        features = features.view(
            features.size(0), -1, features.size(-1))  # (batch_size, 49, 2048)
        return features

    @staticmethod
    def preprocess_images(images: list[Image]) -> Tensor:
        transform = Compose([
            Resize((224, 224)),  # ResNet expects images of 224x224
            ToTensor(),  # Converts PIL image to a tensor with values [0, 1]
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return stack([transform(image.convert("RGB")) for image in images])

    def _get_passive_resnet(self) -> Sequential:
        """ Prepare the ResNet-50 model by setting it to evaluation mode, freezing parameters, and removing the final layers to retain spatial information in the output.

        Returns:
            Sequential: Modified ResNet-50 model as a Sequential module.
        """
        resnet: ResNet = resnet50(weights='ResNet50_Weights.DEFAULT').eval()
        for param in resnet.parameters():
            param.requires_grad_(False)
        return Sequential(*list(resnet.children())[:-2])
