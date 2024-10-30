from nltk.translate.bleu_score import corpus_bleu
from torch import no_grad, device, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

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
        data (DataLoader): DataLoader providing images and reference captions.
        device (torch.device): Device to run evaluation on (default: CPU).

    Returns:
        float: The BLEU score for the evaluated dataset.
    """
    model.eval()
    model.to(device)

    references, candidates = [], []

    with no_grad():
        # Progress bar for evaluation batches
        with tqdm(total=len(data), desc="Evaluating", unit="batch") as pbar:
            for images, captions in data:
                images = images.to(device)
                predictions = model(images)

                # Prepare reference and candidate sentences for BLEU scoring
                for index, prediction in enumerate(predictions):
                    references.append([dataset.vocabulary.tokenize(
                        dataset.tensor_to_caption(captions[index]))])
                    candidates.append(dataset.vocabulary.tokenize(
                        dataset.tensor_to_caption(prediction.argmax(dim=1))))

                pbar.update(1)

    # Calculate final BLEU score
    bleu = corpus_bleu(references, candidates)
    print(f"[Evaluation] Final BLEU Score: {bleu:.4f}")
    return bleu
