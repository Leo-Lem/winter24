import os
from torch.utils.data import Dataset, random_split
from PIL import Image
from pandas import read_csv
from torch import tensor, Tensor
from torchvision.transforms import Compose

from .vocabulary import Vocabulary


class FlickrDataset(Dataset):
    """ Flickr dataset for image captioning. """

    def __init__(self,
                 image_path: str,
                 images: list[str],
                 captions: list[str],
                 vocabulary: Vocabulary,
                 image_transform: Compose):
        """
        Args:
            image_path (str): Path to the folder containing the images.
            images (list[str]): List of image filenames.
            captions (list[str]): List of captions for the images.
            vocabulary (Vocabulary): Vocabulary object for the captions.
            transform_images (Compose): Transform to apply to images.
        """
        self.image_path = image_path
        self.images = images
        self.captions = captions
        self.vocabulary = vocabulary
        self.image_transform = image_transform

    def __len__(self) -> int: return len(self.captions)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.get_image_tensor(index), self.get_caption(index)

    def get_image_tensor(self, index: int) -> Tensor:
        """ Get the image tensor for the given index. """
        return self.get_image_tensor_at_path(os.path.join(self.image_path, "Images", self.images[index]))

    def get_image_tensor_at_path(self, image_path: str) -> Tensor:
        """ Get the image tensor for the given path. """
        image = Image.open(image_path).convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)
        return image

    def get_caption(self, index: int) -> str:
        """ Get the encoded caption for the given index. """
        raw = self.captions[index]
        encoded = [self.vocabulary.token_to_id["<SOS>"]] + \
            self.vocabulary.numericalize(raw) + \
            [self.vocabulary.token_to_id["<EOS>"]]
        return tensor(encoded)

    def split(self, train_size: float = 0.8) -> tuple["FlickrDataset", "FlickrDataset"]:
        """ Split the dataset into a training and validation set."""

        subsets = random_split(self, [train_size, 1 - train_size])
        return (FlickrDataset(self.image_path,
                              [self.images[i] for i in subsets[0].indices],
                              [self.captions[i] for i in subsets[0].indices],
                              self.vocabulary,
                              self.image_transform),
                FlickrDataset(self.image_path,
                              [self.images[i] for i in subsets[1].indices],
                              [self.captions[i] for i in subsets[1].indices],
                              self.vocabulary,
                              self.image_transform))


class LoadedFlickrDataset(FlickrDataset):
    """ Flickr dataset for image captioning with captions loaded from file. """

    def __init__(self,
                 path: str,
                 captions_file: str = "captions.csv",
                 num_captions: int = None,
                 image_transform: Compose = None,
                 vocabulary_threshold: int = 5):
        """
        Args:
            path (str): Path to the dataset folder.
            captions_file (str): Name of the file containing the captions.
            num_images (int): Number of captions to load.
            image_transform (Compose): Transform to apply to images.
            vocabulary_threshold (int): Minimum number of occurrences for a word to be included in the vocabulary.
        """
        images_with_captions = read_csv(os.path.join(path, captions_file),
                                        nrows=num_captions)
        captions = images_with_captions["caption"].tolist()
        images = images_with_captions["image"].tolist()
        vocab = Vocabulary(captions, threshold=vocabulary_threshold)

        super().__init__(
            image_path=path,
            images=images,
            captions=captions,
            vocabulary=vocab,
            image_transform=image_transform
        )
