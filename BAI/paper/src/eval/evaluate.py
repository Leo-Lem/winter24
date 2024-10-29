from torch.nn import Module
from nltk.translate import bleu_score

from data import FlickrDataset


def evaluate(model: Module, data: FlickrDataset):
    references = []
    candidates = []
    for i in range(len(data)):
        reference = data[i][1]
        candidate = model.predict(data[i][0])
        references.append(reference)
        candidates.append(candidate)
    return bleu_score.corpus_bleu(references, candidates)

# def validateModel(self,
#                       batch_size: int = None,
#                       num_workers: int = None,
#                       device: str = "cpu") -> float:
#         """ Validate the model on the validation dataset.
#         Args:
#             batch_size: The number of samples in each batch (defaults to the model's batch size).
#             num_workers: The number of subprocesses to use for data loading (defaults to the model's num_workers).
#         Returns:
#             The average loss on the validation dataset.
#         """

#         loader = self._validation_loader(
#             batch_size=batch_size, num_workers=num_workers)

#         total_loss = 0

#         self.eval()
#         for idx, (images, captions) in enumerate(loader):
#             print(f"[Validation] Batch {idx+1}/{len(loader)}")

#             images = images.to(device)
#             captions = captions.to(device)

#             with no_grad():
#                 outputs, _ = self(images, captions)

#             targets = captions[:, 1:]
#             loss: CrossEntropyLoss = self.criterion(
#                 outputs.to(device).view(-1, len(loader.dataset.vocabulary)), targets.reshape(-1))

#             total_loss += loss.item()

#         loss = total_loss / len(loader)
#         print(f"\n[Validation] Loss: {loss}")
#         return loss
