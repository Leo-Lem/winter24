from torch import Tensor, cat
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .vocabulary import Vocabulary


def loader(dataset: Dataset, batch_size: int = 5) -> DataLoader:
    class CollateAndPadBatch:
        """ Collate and pad batch of images and captions. """

        def __call__(self, batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:

            images = [item[0].unsqueeze(0) for item in batch]
            images = cat(images, dim=0)

            captions = [item[1] for item in batch]
            captions = pad_sequence(captions,
                                    batch_first=True,
                                    padding_value=Vocabulary.pad_index)

            return images, captions

    return DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=True, collate_fn=CollateAndPadBatch())
