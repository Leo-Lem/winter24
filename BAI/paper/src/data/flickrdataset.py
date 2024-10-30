import os
from torch.utils.data import Dataset, random_split
from PIL import Image
from pandas import read_csv
from torch import Tensor
from collections import Counter
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from .vocabulary import Vocabulary


class FlickrDataset(Dataset):
    """ Flickr dataset for image captioning. """

    def __init__(self,
                 path: str,
                 images_folder: str = "Images",
                 captions_file: str = "captions.csv",
                 num_captions: int = None,
                 image_transform: Compose = Compose([
                     Resize((224, 224)),
                     ToTensor(),
                     Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
                 ]),
                 vocabulary_threshold: int = 5):
        """
        Args:
            path (str): Path to the dataset folder.
            captions_file (str): Name of the file containing the captions.
            num_images (int): Number of captions to load.
        """
        self.image_path = os.path.join(path, images_folder)
        images_with_captions = read_csv(
            os.path.join(path, captions_file), nrows=num_captions)
        self.captions = images_with_captions["caption"].tolist()
        self.images = images_with_captions["image"].tolist()
        self.image_transform = image_transform
        self.vocabulary = Vocabulary(
            self.captions, threshold=vocabulary_threshold)

    def __len__(self) -> int: return len(self.captions)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image = self.image_to_tensor(
            Image.open(os.path.join(self.image_path, self.images[index])))
        caption = self.caption_to_tensor(self.captions[index])
        return image, caption

    def image_to_tensor(self, image: Image.Image) -> Tensor:
        return self.image_transform(image)

    def caption_to_tensor(self, caption: str) -> Tensor:
        return Tensor([self.vocabulary.sos_index, *self.vocabulary.numericalize(caption), self.vocabulary.eos_index])

    def tensor_to_caption(self, tensor: Tensor) -> str:
        return self.vocabulary.denumericalize(tensor.tolist()[1:-1])
