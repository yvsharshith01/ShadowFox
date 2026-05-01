import torch
from sklearn.metrics import classification_report
from model import get_model
from utils import get_data_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, val_loader = get_data_loaders("data")

model = get_model().to(device)
model.load_state_dict(torch.load("models/model.pth"))
model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print(classification_report(all_labels, all_preds))