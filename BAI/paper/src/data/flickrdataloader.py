
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import cat, Tensor

from .flickrdataset import FlickrDataset


class FlickrDataloader(DataLoader):
    """ Load Flickr dataset batches. """

    def __init__(self, dataset: FlickrDataset, batch_size: int = 5, num_workers: int = 2, shuffle: bool = True):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=FlickrDataloader.BatchPadding(pad_token_index=dataset.vocabulary.token_to_id["<PAD>"],
                                                     batch_dimension_first=True)
        )

    class BatchPadding:
        """ Apply padding to  """

        def __init__(self, pad_token_index: int, batch_dimension_first: bool = False):
            """
            Args:
                pad_token_index: The index of the padding token in the vocabulary.
                batch_dimension_first: Whether the batch dimension is the first dimension.
            """
            self.pad_index = pad_token_index
            self.batch_first = batch_dimension_first

        def __call__(self, batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
            """ Combine images and captions of a batch into a single tensor each and pad the captions.

            Args:
                batch: The batch of caption and image tensors.
            """
            images = [item[0].unsqueeze(0) for item in batch]
            images = cat(images, dim=0)

            captions = [item[1] for item in batch]
            captions = pad_sequence(captions,
                                    batch_first=self.batch_first,
                                    padding_value=self.pad_index)

            return images, captions
