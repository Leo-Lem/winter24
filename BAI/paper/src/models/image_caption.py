from torch import Tensor, save, no_grad, load
from torch.nn import Module, CrossEntropyLoss
from torchvision.transforms import Compose
from torch.optim import Adam
from PIL import Image


from data import FlickrDataset, FlickrDataloader
from .encoder import ImageEncoder
from .decoder import CaptionDecoder


class ImageCaption(Module):
    def __init__(self,
                 dataset: FlickrDataset,
                 encoder: ImageEncoder,
                 decoder: CaptionDecoder,
                 learning_rate: float,
                 batch_size: int = 5,
                 num_workers: int = 2,
                 loss_fn: CrossEntropyLoss = None,
                 model_path: str = "imagecaption.state"):
        super(ImageCaption, self).__init__()
        self.model_path = model_path

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.encoder = encoder
        self.decoder = decoder

        self.optimizer = Adam(self.parameters(), lr=learning_rate)
        self.criterion = loss_fn if loss_fn else CrossEntropyLoss(
            ignore_index=dataset.vocabulary.token_to_id["<PAD>"])

    def forward(self, images: Tensor, captions: Tensor) -> Tensor:
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def trainModel(self,
                   epochs: int = 5,
                   batch_size: int = None,
                   num_workers: int = None,
                   resume: bool = False,
                   device: str = "cpu"):
        """" Train the model for a given number of epochs.
        Args:
            batch_size: The number of samples in each batch (defaults to the model's batch size).
            num_workers: The number of subprocesses to use for data loading (defaults to the model's num_workers).
            epochs: The number of times to iterate over the training dataset.
            resume: Whether to resume training from the last checkpoint.
        """

        start_epoch = self._load() if resume else 0
        if start_epoch == epochs:
            print("[Training] Model already trained.")
            return
        elif start_epoch > epochs:
            raise ValueError(
                f"Model has already been trained for {start_epoch} epochs.")
        elif start_epoch > 0:
            print(f"[Training] Resuming from epoch {start_epoch}.")

        loader = self._training_loader(batch_size, num_workers)

        self.train()
        for epoch in range(start_epoch, epochs):
            for idx, (images, captions) in enumerate(loader):
                print(
                    f"[Training | Epoch {epoch+1}/{epochs}] Batch {idx+1}/{len(loader)}")

                images = images.to(device)
                captions = captions.to(device)

                # Zero the gradients.
                self.optimizer.zero_grad()

                # Feed forward
                outputs, _ = self(images, captions)

                # Calculate the batch loss.
                targets = captions[:, 1:]
                loss = self.criterion(
                    outputs.to(device).view(-1, len(loader.dataset.vocabulary)), targets.reshape(-1))

                # Backward pass.
                loss.backward()

                # Update the parameters in the optimizer.
                self.optimizer.step()

            self._save(epoch+1)
            print(f"\n[Training | Epoch {epoch+1}/{epochs}] Loss: {loss}\n")

    def validateModel(self,
                      batch_size: int = None,
                      num_workers: int = None,
                      device: str = "cpu") -> float:
        """ Validate the model on the validation dataset.
        Args:
            batch_size: The number of samples in each batch (defaults to the model's batch size).
            num_workers: The number of subprocesses to use for data loading (defaults to the model's num_workers).
        Returns:
            The average loss on the validation dataset.
        """

        loader = self._validation_loader(
            batch_size=batch_size, num_workers=num_workers)

        total_loss = 0

        self.eval()
        for idx, (images, captions) in enumerate(loader):
            print(f"[Validation] Batch {idx+1}/{len(loader)}")

            images = images.to(device)
            captions = captions.to(device)

            with no_grad():
                outputs, _ = self(images, captions)

            targets = captions[:, 1:]
            loss: CrossEntropyLoss = self.criterion(
                outputs.to(device).view(-1, len(loader.dataset.vocabulary)), targets.reshape(-1))

            total_loss += loss.item()

        loss = total_loss / len(loader)
        print(f"\n[Validation] Loss: {loss}")
        return loss

    def predict(self, image_path: str, transform: Compose, max_len: int = 20) -> tuple[str, Tensor]:
        """
        Generate a caption for a given image.

        Args:
            image_path: Path to the image file.
            transform: Preprocessing transforms for the image.
            max_len: Maximum length of the generated caption.

        Returns:
            Generated caption as a string and the image tensor.
        """
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)  # Add batch dimension

        self.eval()
        with no_grad():
            features = self.encoder(image)
            caption, _ = self.decoder.generate_caption(
                features, self.dataset.vocabulary, max_len=max_len)

        return " ".join(caption), image.squeeze(0)

    def _save(self, epoch: int):
        """ Save the model to a checkpoint. """
        save({
            "epoch": epoch,
            "state": self.state_dict()
        }, self.model_path)

    def _load(self) -> int:
        """ Load the model from the last checkpoint. """
        try:
            checkpoint = load(self.model_path, weights_only=True)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['state'].items()
                               if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)

            self.load_state_dict(model_dict)
            print(
                f"[Load] From {self.model_path} at epoch {checkpoint['epoch']}.")
            return checkpoint['epoch']
        except FileNotFoundError:
            return 0

    def _training_loader(self, batch_size: int, num_workers: int) -> FlickrDataloader:
        return self._loaders(batch_size, num_workers)[0]

    def _validation_loader(self, batch_size: int, num_workers: int) -> FlickrDataloader:
        return self._loaders(batch_size, num_workers)[1]

    def _loaders(self, batch_size: int, num_workers: int) -> tuple[FlickrDataloader, FlickrDataloader]:
        batch_size = batch_size if batch_size else self.batch_size
        num_workers = num_workers if num_workers else self.num_workers

        training, validation = self.dataset.split(train_size=.8)
        return (FlickrDataloader(training, batch_size=batch_size, num_workers=num_workers),
                FlickrDataloader(validation, batch_size=batch_size, num_workers=num_workers))
