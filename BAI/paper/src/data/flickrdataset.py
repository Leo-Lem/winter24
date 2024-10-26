from data.vocabulary import Vocabulary
import os
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pandas import read_csv
from torch import tensor, Tensor, cat
from torchvision.transforms import Compose


class FlickrDataset(Dataset):
    """ Flickr dataset for image captioning. """

    def __init__(self, path: str, captions: str = "captions.txt", transform: Compose = None, threshold: int = 5):
        self.path = path
        self.images_with_captions = read_csv(os.path.join(self.path, captions))
        self.transform = transform
        self.vocab = Vocabulary(threshold, self.captions)

    def __len__(self) -> int: return len(self.captions)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        raw_caption = self.captions[index]
        numericalized_caption = [self.vocab.token_to_id["<SOS>"]] + \
            self.vocab.numericalize(raw_caption) + \
            [self.vocab.token_to_id["<EOS>"]]
        caption = tensor(numericalized_caption)

        image_path = os.path.join(self.path, "Images", self.images[index])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, caption

    @property
    def images(self) -> list[str]:
        return self.images_with_captions["image"].tolist()

    @property
    def captions(self) -> list[str]:
        return self.images_with_captions["caption"].tolist()

    def data_loader(self, batch_size: int = 4, num_workers: int = 1, shuffle: bool = True):
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=FlickrDataset.CapsCollate(
                pad_index=self.vocab.token_to_id["<PAD>"], batch_first=True)
        )

    class CapsCollate:
        """ Collate to apply the padding to the captions with dataloader. """

        def __init__(self, pad_index: int, batch_first: bool = False):
            self.pad_index = pad_index
            self.batch_first = batch_first

        def __call__(self, batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
            """ Apply padding to the captions. """
            images = [item[0].unsqueeze(0) for item in batch]
            images = cat(images, dim=0)

            targets = [item[1] for item in batch]
            targets = pad_sequence(
                targets, batch_first=self.batch_first, padding_value=self.pad_index)

            return images, targets
