from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torchvision.datasets import ImageFolder

class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception:
            print(f"Skipping corrupted image at index {index}")
            return self.__getitem__((index + 1) % len(self))

def get_data_loaders(data_dir, batch_size=32):

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    train_data = SafeImageFolder("data/train", transform=train_transforms)
    val_data   = SafeImageFolder("data/val", transform=val_transforms)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=32)


    return train_loader, val_loader
