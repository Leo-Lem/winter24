from torch import cuda, device
from torch.utils.data import random_split, DataLoader

from data import FlickrDataset
from models import ImageCaption, ResnetImageEncoder, GRUCaptionDecoder
from eval import train, evaluate

# DIR = "/content/drive/MyDrive"
DIR = "."
CAPTION_LIMIT = None
DEVICE = device("cuda" if cuda.is_available() else "cpu")
BATCH_SIZE = 800 if cuda.is_available() else 400
NUM_WORKERS = 2 if cuda.is_available() else 0
PIN_MEMORY = cuda.is_available()

# --- Data ---
dataset = FlickrDataset(path=f"{DIR}/flickr8k", caption_limit=CAPTION_LIMIT)
train_dataset, eval_dataset = random_split(dataset, [.8, .2])


# --- Model ---
encoder = ResnetImageEncoder()
decoder = GRUCaptionDecoder(vocabulary_size=len(dataset.vocabulary))
model = ImageCaption(encoder=encoder, decoder=decoder)


# --- Training ---

train(model,
      data=DataLoader(train_dataset,
                      batch_size=BATCH_SIZE,
                      num_workers=NUM_WORKERS,
                      shuffle=True,
                      pin_memory=PIN_MEMORY),
      epochs=5,
      device=DEVICE)

# --- Evaluation ---
evaluate(model,
         data=DataLoader(eval_dataset,
                         batch_size=BATCH_SIZE,
                         num_workers=NUM_WORKERS,
                         shuffle=True,
                         pin_memory=PIN_MEMORY),
         dataset=dataset,
         device=DEVICE)
