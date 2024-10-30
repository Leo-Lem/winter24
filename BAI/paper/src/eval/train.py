from torch import device, Tensor
from torch.optim import Adam, Optimizer
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

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

    print("[Training] Starting...")
    for epoch in range(checkpoint, epochs):
        print(f"[Epoch {epoch + 1}/{epochs}]")
        epoch_loss = 0

        for batch, (images, captions) in enumerate(data):
            assert images.size(0) == captions.size(0), \
                f"Batch size mismatch: {images.size(0)} != {captions.size(0)}"

            images: Tensor = images.to(device)
            captions: Tensor = captions.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs: Tensor = model(images, max_len=captions.size(1))

            assert outputs.size(0) == captions.size(0) and outputs.size(1) == captions.size(1), \
                f"Output shape mismatch: {outputs.size()} != {captions.size()}"

            # Skip <SOS> token in targets, align outputs and targets, flatten for loss
            targets = captions[:, 1:].reshape(-1).long()
            outputs = outputs[:, :-1].reshape(-1, outputs.size(-1)).float()

            assert outputs.requires_grad, "Outputs must require gradients."

            assert outputs.size(0) == targets.size(0), \
                f"Batch size mismatch: {outputs.size(0)} != {targets.size(0)}"

            # Compute loss
            assert outputs.dim() == 2 and targets.dim() == 1, \
                "Output and target shapes are not compatible with CrossEntropyLoss"
            loss: Tensor = criterion(outputs, targets)
            epoch_loss += loss.item()

            # Backward and optimize
            loss.backward()
            optimizer.step()

            print(
                f"\t[Epoch {epoch + 1}/{epochs}] Batch {batch + 1}/{len(data)} - Loss: {loss.item():.4f}")

        print(
            f"[Epoch {epoch + 1}] Average Loss: {epoch_loss / len(data):.4f}")

        model.save_checkpoint(epoch + 1)

    print("[Training] Training completed.")
    return model
