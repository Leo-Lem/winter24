from torch import cuda, device
from torchvision import transforms as T

from data import LoadedFlickrDataset, Visualiser
from models import ImageCaption, ImageEncoder, CaptionDecoder

DIR = "/content/drive/MyDrive"
# DIR = "."
DEVICE = device("cuda" if cuda.is_available() else "cpu")

# --- Dataset ---
transforms = T.Compose([
    T.Resize(226),
    T.RandomCrop(224),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
dataset = LoadedFlickrDataset(
    f"{DIR}/flickr8k",
    num_captions=None,
    image_transform=transforms
)

# --- Model ---
encoder = ImageEncoder(
).to(DEVICE)
decoder = CaptionDecoder(
    embed_size=300,
    vocab_size=len(dataset.vocabulary),
    attention_dim=256,
    encoder_dim=2048,
    decoder_dim=512
).to(DEVICE)
model = ImageCaption(
    dataset=dataset,
    encoder=encoder,
    decoder=decoder,
    batch_size=5,
    num_workers=2,
    learning_rate=3e-4
).to(DEVICE)

model.trainModel(epochs=2, resume=True, device=DEVICE)
model.validateModel(device=DEVICE)

# --- Inference ---
visualiser = Visualiser()

test_image = f"{DIR}/image.jpg"
prediction = model.predict(test_image, transforms)
visualiser.display_image(prediction[1], prediction[0])
