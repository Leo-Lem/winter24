from torch import Tensor
from matplotlib import pyplot as plt


class Visualiser:
    def __init__(self):
        plt.ion()

    def display_image(self, tensor: Tensor, title: str):
        """ Show image with caption and pause so that plots are updated. """
        plt.imshow(self._unnormalize(tensor))
        if title is not None:
            plt.title(title)
        plt.pause(.001)

    def _unnormalize(self, tensor: Tensor):
        tensor = tensor.clone()  # Avoid modifying the original tensor in-place

        # Apply the un-normalization (assuming ImageNet normalization was applied)
        tensor[0] = tensor[0] * 0.229 + 0.485
        tensor[1] = tensor[1] * 0.224 + 0.456
        tensor[2] = tensor[2] * 0.225 + 0.406

        # Convert from [C, H, W] to [H, W, C] for displaying with matplotlib
        return tensor.permute(1, 2, 0).numpy()
