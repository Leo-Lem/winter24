from nltk.translate.bleu_score import corpus_bleu
from torch import no_grad, device, Tensor
from torch.utils.data import DataLoader

from models import ImageCaption
from data import FlickrDataset


def evaluate(model: ImageCaption,
             data: DataLoader,
             dataset: FlickrDataset,
             device: device = device("cpu")) -> float:
    """
    Evaluate the ImageCaption model on the provided dataset using BLEU score.

    Args:
        model (ImageCaption): The image captioning model to evaluate.
        dataloader (DataLoader): DataLoader providing images and reference captions.
        device (torch.device): Device to run evaluation on (default: CPU).

    Returns:
        float: The BLEU score for the evaluated dataset.
    """
    model.eval()
    model.to(device)

    references = []
    candidates = []

    with no_grad():
        print("[Evaluation] Evaluating...")
        for batch, (images, captions) in enumerate(data):
            images: Tensor = images.to(device)
            predictions: Tensor = model(images)

            # Prepare reference and candidate sentences for BLEU scoring
            for index, prediction in enumerate(predictions):
                references.append(dataset.vocabulary
                                  .tokenize(dataset.tensor_to_caption(captions[index])))
                candidates.append(dataset.vocabulary
                                  .tokenize(dataset.tensor_to_caption(prediction.argmax(dim=1))))

            # Calculate batch-level BLEU score for tracking
            batch_bleu = corpus_bleu(references,
                                     candidates,
                                     weights=(0.25, 0.25, 0.25, 0.25))
            print(
                f"\t[Evaluation] Batch {batch + 1}/{len(data)} - BLEU Score: {batch_bleu:.4f}")

    bleu = corpus_bleu(references, candidates)
    print(f"[Evaluation] Evaluation compelted. Final BLEU Score: {bleu:.4f}")
    return bleu
