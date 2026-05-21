import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
import matplotlib.pyplot as plt

import json
import os

import yaml

import boto3

from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(4)
print(f'Using device: {device}')

# =========================================================
# S3 DOWNLOAD
# =========================================================

s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:9000",
    aws_access_key_id="minioadmin",
    aws_secret_access_key="minioadmin"
)

s3.download_file(
    "models",
    "model_stage1.pth",
    "artifacts/model_stage1.pth"
)

# Params
with open("params.yaml") as f:
    params = yaml.safe_load(f)

lr = params["retrain"]["lr"]
batch_size = params["retrain"]["batch_size"]
epochs = params["retrain"]["epochs"]

# =========================================================
# DATASET
# =========================================================

class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.root_dir / self.data.iloc[idx]['file_name']
        image = Image.open(img_path).convert('RGB')
        label = self.data.iloc[idx]['label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# =========================================================
# DATA
# =========================================================

train_dataset = ImageDataset(
    csv_file="ai-vs-human-generated-dataset-hw/Train_2/train.csv",
    root_dir="ai-vs-human-generated-dataset-hw/Train_2",
    transform=train_transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

print(f'Train dataset size: {len(train_dataset)}')

# =========================================================
# MODEL
# =========================================================

model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model.load_state_dict(
    torch.load("artifacts/model_stage1.pth")
)

model = model.to(device)
#print(f'Model architecture:')
#print(model)

# =========================================================
# TRAIN CONFIG
# =========================================================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# TENSORBOARD

writer = SummaryWriter("my_logs/retrain")

hparams = {
    "epochs": epochs,
    "learning_rate": lr,
    "batch_size": batch_size,
    "dataset": "Train_2",
    "test dataset": "Test_2",
    'model': 'ResNet18'
}

with open("artifacts/retrain_hparams.json", "w") as f:
    json.dump(hparams, f, indent=4)

# TRAIN FUNCTION

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(dataloader, desc='Training'):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Подсчет метрик
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss, epoch_acc, epoch_f1

# =========================================================
# TRAIN LOOP
# =========================================================
import time

num_epochs = epochs
train_losses = []
train_accs = []
train_f1s = []

# Логгирование метрик для tensorboard
start = time.time()

for epoch in range(num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    print('-' * 50)
    
    train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
    scheduler.step()
    
    print("Epoch time:", time.time() - start)
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    train_f1s.append(train_f1)
    
    writer.add_scalar('Loss/Retrain', train_loss, epoch)
    writer.add_scalar('Accuracy/Retrain', train_acc, epoch)
    writer.add_scalar('F1/Retrain', train_f1, epoch)
    writer.add_scalar('Learning_Rate/Retrain', scheduler.get_last_lr()[0], epoch)
    
    print(f'Retrain Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}')

print('\nTraining completed!')

writer.add_hparams(
    hparams,
    {
        "hparam/test_accuracy": train_acc,
        "hparam/test_f1": train_f1,
        "hparam/test_loss": train_loss
    }
)

# =========================================================
# SAVE MODEL
# =========================================================

torch.save(
    model.state_dict(),
    "artifacts/model_dvc.pth"
)

writer.close()

print("Retrain completed")











