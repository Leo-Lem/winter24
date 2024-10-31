import os
from torch import tensor, Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import pickle
from tqdm import tqdm
import pandas as pd


from .vocabulary import Vocabulary


class FlickrDataset(Dataset):
    def __init__(self,
                 path: str,
                 caption_limit: int = None,
                 images_folder: str = "Images",
                 captions_file: str = "captions.csv",
                 image_transform: Compose = None,
                 data_file: str = "compressed_data.pkl",
                 vocabulary_threshold: int = 5):
        self.image_path = os.path.join(path, images_folder)
        self.captions_file = os.path.join(path, captions_file)
        self.data_path = os.path.join(path, data_file)
        self.caption_limit = caption_limit
        self.image_transform = image_transform or Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if os.path.exists(self.data_path) and caption_limit is None:
            print("Loading preprocessed data...")
            self.images, \
                self.captions, \
                self.vocabulary, \
                self.caption_to_image_index = self._load()
        else:
            self.images, \
                self.captions, \
                self.vocabulary, \
                self.caption_to_image_index = self._process(
                    vocabulary_threshold)
            print("Saving preprocessed data...")
            self._save(self.images,
                       self.captions,
                       self.vocabulary,
                       self.caption_to_image_index)

    def _save(self,
              images: list[Tensor],
              captions: list[Tensor],
              vocabulary: Vocabulary,
              caption_to_image_index: dict[int, int]):
        with open(self.data_path, 'wb') as f:
            pickle.dump({
                'images': images,
                'captions': captions,
                'vocabulary': vocabulary,
                'caption_to_image_index': caption_to_image_index
            }, f)

    def _load(self) -> tuple[list[Tensor], list[Tensor], Vocabulary, dict[int, int]]:
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
            return data['images'], data['captions'], data['vocabulary'], data['caption_to_image_index']

    def _process(self, vocabulary_threshold: int) -> tuple[list[Tensor], list[Tensor], Vocabulary, dict[int, int]]:
        data = pd.read_csv(self.captions_file, nrows=self.caption_limit)

        captions = data['caption'].tolist()
        vocabulary = Vocabulary(captions, threshold=vocabulary_threshold)

        images = data['image'].tolist()
        unique_images: list = data['image'].unique().tolist()
        image_tensors = []
        for img_name in tqdm(unique_images, desc="Preprocessing Images", unit="images"):
            image_path = os.path.join(self.image_path, img_name)
            image = Image.open(image_path).convert('RGB')
            image_tensors.append(self.image_transform(image))

        caption_tensors = []
        caption_to_image_index = {}
        for index, caption in enumerate(tqdm(captions, desc="Tokenizing Captions", unit="captions")):
            caption_tensors.append(tensor(vocabulary.numericalize(caption)))
            caption_to_image_index[index] = unique_images.index(images[index])

        return image_tensors, caption_tensors, vocabulary, caption_to_image_index

    def __len__(self): return len(self.captions)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return tensor(self.captions[index]), tensor(self.images[self.caption_to_image_index[index]])

    def tensor_to_caption(self, tensor: Tensor) -> str:
        return self.vocabulary.denumericalize(tensor.tolist()[1:-1])

    def image_to_tensor(self, image: Image) -> Tensor:
        return self.image_transform(image)
