import random
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

# Load datasets
PLANTVILLAGE_DIR = Path("../PlantVillage-Dataset/raw/color")
PLANTDOC_TRAIN_DIR = Path("../PlantDoc-Dataset/train")
PLANTDOC_TEST_DIR = Path("../PlantDoc-Dataset/test")

class PlantDiseaseDataset(Dataset):
    def __init__(self, samples, label2idx, transform=None):
        self.samples = samples
        self.label2idx = label2idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_name = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label_idx = self.label2idx[label_name]
        return img, torch.tensor(label_idx, dtype=torch.long)
    
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

