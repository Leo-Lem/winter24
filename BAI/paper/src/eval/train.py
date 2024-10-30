from torch import device, Tensor
from torch.optim import Adam, Optimizer
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import ImageCaption
from data import Vocabulary


def train(model: ImageCaption,
          data: DataLoader,
          epochs: int = 5,
          device: device = device("cpu"),
          optimizer: Optimizer = None,
          criterion: CrossEntropyLoss = None):
    """
    Train the ImageCaption model.

    Args:
        model (ImageCaption): The caption generation model to train.
        data (DataLoader): Dataloader providing the training data.
        epochs (int): Number of epochs to train for (default: 5).
        device (device): Device to use for training (default: CPU).
        optimizer (Optimizer): Optimizer for training (default: Adam).
        criterion (CrossEntropyLoss): Loss function (default: CrossEntropyLoss).

    Returns:
        ImageCaption: The trained model with the latest checkpoint.
    """
    model.to(device)
    model.train()

    optimizer = Adam(model.parameters(),
                     lr=1e-3) if optimizer is None else optimizer
    criterion = CrossEntropyLoss(ignore_index=Vocabulary.pad_index).to(
        device) if criterion is None else criterion

    checkpoint = model.load_checkpoint()

    for epoch in range(checkpoint, epochs):
        print(f"[Epoch {epoch + 1}/{epochs}]")
        epoch_loss = 0

        # Progress bar for each batch
        with tqdm(total=len(data), desc=f"Training Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for images, captions in data:
                images, captions = images.to(device), captions.to(device)
                optimizer.zero_grad()

                # Forward pass
                outputs = model(images, max_len=captions.size(1))
                loss = criterion(
                    outputs.view(-1, outputs.size(-1)), captions.view(-1))

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({"Batch Loss": loss.item()})
                pbar.update(1)

        print(
            f"[Epoch {epoch + 1}] Average Loss: {epoch_loss / len(data):.4f}")

        model.save_checkpoint(epoch + 1)

    return model
