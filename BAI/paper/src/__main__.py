from torch import cuda
from torchvision import transforms as T

from .models import ImageCaption

model = ImageCaption(
    "flickr8k",
    batch_size=4,
    num_workers=2,
    embed_size=300,
    attention_dim=256,
    encoder_dim=2048,
    decoder_dim=512,
    learning_rate=3e-4,
    transform=T.Compose([
        T.Resize(226),
        T.RandomCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),
    device="cuda" if cuda.is_available() else "cpu"
)

model.train(epochs=3)

print(model.predict(model.dataset[0][0]))
