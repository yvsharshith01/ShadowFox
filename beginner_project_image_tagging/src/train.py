import torch
import torch.nn as nn
import torch.optim as optim
from model import get_model
from utils import get_data_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader = get_data_loaders("data")

model = get_model().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

epochs = 10

for epoch in range(epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {train_loss/len(train_loader):.4f}")
    print(f"Train Accuracy: {train_acc:.4f}")

torch.save(model.state_dict(), "models/model.pth")
print("\n Model saved at models/model.pth")