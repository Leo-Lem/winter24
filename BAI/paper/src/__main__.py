from torch import cuda, device, no_grad
from torch.utils.data import random_split
from PIL import Image

from data import FlickrDataset, Vocabulary, loader
from models import ImageCaption, ResnetImageEncoder, GRUCaptionDecoder
from eval import train, evaluate, visualise

# DIR = "/content/drive/MyDrive"
DIR = "."
DEVICE = device("cuda" if cuda.is_available() else "cpu")

# --- Data ---
dataset = FlickrDataset(path=f"{DIR}/flickr8k",
                        num_captions=100)
vocabulary = Vocabulary(dataset.captions,
                        threshold=5)
train_dataset, eval_dataset = random_split(dataset, [.8, .2])


# --- Model ---
encoder = ResnetImageEncoder()
decoder = GRUCaptionDecoder(vocabulary_size=len(vocabulary))
model = ImageCaption(encoder=encoder, decoder=decoder)


# --- Training ---

train(model,
      data=loader(train_dataset),
      epochs=10,
      device=DEVICE)


# --- Evaluation ---
evaluate(model,
         data=loader(eval_dataset),
         decode_caption=dataset.tensor_to_caption,
         device=DEVICE)


# --- Inference ---
with no_grad():
    model.eval()
    image = dataset.image_to_tensor(
        Image.open(f"{DIR}/image.jpg").convert("RGB")).unsqueeze(0).to(DEVICE)
    caption = dataset.tensor_to_caption(model(image))
visualise(image, caption)
