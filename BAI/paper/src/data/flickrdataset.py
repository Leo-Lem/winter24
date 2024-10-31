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
                 images_folder: str = "Images",
                 captions_file: str = "captions.csv",
                 num_captions: int = None,
                 image_transform: Compose = None,
                 data_file: str = "data.pkl",
                 vocabulary_threshold: int = 5):
        self.image_path = os.path.join(path, images_folder)
        self.captions_file = os.path.join(path, captions_file)
        self.data_path = os.path.join(path, data_file)

        self.image_transform = image_transform or Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if os.path.exists(self.data_path) and num_captions is None:
            print("[Dataset] Loading preprocessed data...")
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
                self.images, self.captions, self.vocabulary = \
                    data['images'], data['captions'], data['vocabulary']
        else:
            print("[Dataset] Preprocessing and storing data...")
            self.images, self.captions, self.vocabulary = \
                self._process_and_save(num_captions, vocabulary_threshold)

        print(
            f"[Dataset] {len(self.images)} images and {len(self.captions)} captions are ready.")

    def _process_and_save(self, num_captions, vocabulary_threshold):
        data = pd.read_csv(self.captions_file, nrows=num_captions)
        images, captions = [], []

        # Initialize vocabulary and batch tokenize captions
        raw_captions = data['caption'].tolist()
        vocabulary = Vocabulary(raw_captions, threshold=vocabulary_threshold)

        # Tokenize and numericalize captions in batch
        captions = [tensor(vocabulary.numericalize(caption)) for caption in
                    tqdm(raw_captions, desc="Tokenizing Captions")]

        # Use multiprocessing to process images in parallel
        image_paths = [os.path.join(self.image_path, img_name)
                       for img_name in data['image']]
        images = list(tqdm(map(self.process_image, image_paths),
                           total=len(image_paths), desc="Processing Images"))

        # Save processed data including the vocabulary and numericalized captions
        with open(self.data_path, 'wb') as f:
            pickle.dump({'images': images,
                         'captions': captions,
                         'vocabulary': vocabulary}, f)

        return images, captions, vocabulary

    def process_image(self, image_path):
        """Load and transform an image."""
        image = Image.open(image_path)
        return self.image_transform(image)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        return self.images[index], self.captions[index]

    def tensor_to_caption(self, tensor: Tensor) -> str:
        return self.vocabulary.denumericalize(tensor.tolist()[1:-1])

    def image_to_tensor(self, image: Image) -> Tensor:
        return self.image_transform(image)
