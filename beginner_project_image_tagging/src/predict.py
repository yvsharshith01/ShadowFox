import torch
from PIL import Image
from torchvision import transforms
from model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model().to(device)
model.load_state_dict(torch.load("models/model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def predict_image(path):
    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    output = model(img)
    _, pred = torch.max(output, 1)

    return "Dog" if pred.item() == 1 else "Cat"

import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_path>")
    else:
        image_path = sys.argv[1]
        result = predict_image(image_path)
        print(f"\n Prediction: {result}")