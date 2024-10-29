

#         self.optimizer = Adam(self.parameters(), lr=learning_rate)
#         self.criterion = loss_fn if loss_fn else CrossEntropyLoss(
#             ignore_index=dataset.vocabulary.token_to_id["<PAD>"])
# def trainModel(self,
#                 epochs: int = 5,
#                 batch_size: int = None,
#                 num_workers: int = None,
#                 resume: bool = False,
#                 device: str = "cpu"):
#         """" Train the model for a given number of epochs.
#         Args:
#             batch_size: The number of samples in each batch (defaults to the model's batch size).
#             num_workers: The number of subprocesses to use for data loading (defaults to the model's num_workers).
#             epochs: The number of times to iterate over the training dataset.
#             resume: Whether to resume training from the last checkpoint.
#         """

#         start_epoch = self._load() if resume else 0
#         if start_epoch == epochs:
#             print("[Training] Model already trained.")
#             return
#         elif start_epoch > epochs:
#             raise ValueError(
#                 f"Model has already been trained for {start_epoch} epochs.")
#         elif start_epoch > 0:
#             print(f"[Training] Resuming from epoch {start_epoch}.")

#         loader = self._training_loader(batch_size, num_workers)

#         self.train()
#         for epoch in range(start_epoch, epochs):
#             for idx, (images, captions) in enumerate(loader):
#                 print(
#                     f"[Training | Epoch {epoch+1}/{epochs}] Batch {idx+1}/{len(loader)}")

#                 images = images.to(device)
#                 captions = captions.to(device)

#                 # Zero the gradients.
#                 self.optimizer.zero_grad()

#                 # Feed forward
#                 outputs, _ = self(images, captions)

#                 # Calculate the batch loss.
#                 targets = captions[:, 1:]
#                 loss = self.criterion(
#                     outputs.to(device).view(-1, len(loader.dataset.vocabulary)), targets.reshape(-1))

#                 # Backward pass.
#                 loss.backward()

#                 # Update the parameters in the optimizer.
#                 self.optimizer.step()

#             self._save(epoch+1)
#             print(f"\n[Training | Epoch {epoch+1}/{epochs}] Loss: {loss}\n")
