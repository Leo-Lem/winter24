from torch import cuda, device, no_grad
from torch.utils.data import random_split
from PIL import Image

from data import FlickrDataset, loader
from models import ImageCaption, ResnetImageEncoder, GRUCaptionDecoder
from eval import train, evaluate, visualise

# DIR = "/content/drive/MyDrive"
# CAPTION_LIMIT = None
DIR = "."
CAPTION_LIMIT = 1000
DEVICE = device("cuda" if cuda.is_available() else "cpu")
BATCH_SIZE = 800 if cuda.is_available() else 5
NUM_WORKERS = 2 if cuda.is_available() else 0
PIN_MEMORY = cuda.is_available()

# --- Data ---
dataset = FlickrDataset(path=f"{DIR}/flickr8k", num_captions=CAPTION_LIMIT)
train_dataset, eval_dataset = random_split(dataset, [.8, .2])


# --- Model ---
encoder = ResnetImageEncoder()
decoder = GRUCaptionDecoder(vocabulary_size=len(dataset.vocabulary))
model = ImageCaption(encoder=encoder, decoder=decoder)


# --- Training ---

train(model,
      data=loader(train_dataset,
                  batch_size=BATCH_SIZE,
                  num_workers=NUM_WORKERS,
                  pin_memory=PIN_MEMORY),
      epochs=5,
      device=DEVICE)


# --- Evaluation ---
evaluate(model,
         data=loader(eval_dataset,
                     batch_size=BATCH_SIZE,
                     num_workers=NUM_WORKERS,
                     pin_memory=PIN_MEMORY),
         dataset=dataset,
         device=DEVICE)


# --- Inference ---
with no_grad():
    model.eval()
    image = Image.open(f"{DIR}/image.jpg").convert("RGB")
    caption = dataset.tensor_to_caption(model(
        dataset.image_to_tensor(image).unsqueeze(0).to(DEVICE)))
visualise(image, caption)
