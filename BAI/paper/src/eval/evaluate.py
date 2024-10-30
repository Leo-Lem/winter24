from nltk.translate.bleu_score import corpus_bleu
from torch import no_grad, device, Tensor
from torch.utils.data import DataLoader
from typing import Callable

from models import ImageCaption


def evaluate(model: ImageCaption,
             data: DataLoader,
             decode_caption: Callable[[Tensor], str],
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
        for batch, (images, captions) in enumerate(data):
            images: Tensor = images.to(device)
            predictions: Tensor = model(images)

            # Prepare reference and candidate sentences for BLEU scoring
            for i in range(len(predictions)):
                # Reference: Each reference caption split into tokens (wrapped for BLEU compatibility)
                reference_caption = [decode_caption(captions[i])]
                references.append(reference_caption)

                # Candidate: Predicted caption as a token list
                candidate_caption = decode_caption(predictions[i])
                candidates.append(candidate_caption)

            # Calculate batch-level BLEU score for tracking
            batch_bleu = corpus_bleu(references, candidates)
            print(
                f"[Evaluation] Batch {batch + 1}/{len(data)} - BLEU Score: {batch_bleu:.4f}")

    bleu = corpus_bleu(references, candidates)
    print(f"[Evaluation] Final BLEU Score: {bleu:.4f}")
    return bleu
