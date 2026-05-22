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

# Экспорт параметров из params.yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

lr = params["retrain"]["lr"]
batch_size = params["retrain"]["batch_size"]
epochs = params["retrain"]["epochs"]

# Подготовка данных для валидации

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


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = ImageDataset(
    csv_file='ai-vs-human-generated-dataset-hw/Test_2/test.csv',
    root_dir='ai-vs-human-generated-dataset-hw/Test_2',
    transform=test_transform
)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f'Test dataset size: {len(test_dataset)}')

# Создаем модель и подгружаем дообученное состояние из папки с артефактами

model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model.load_state_dict(
    torch.load("artifacts/model_dvc.pth")
)

model = model.to(device)
#print(f'Model architecture:')
#print(model)

criterion = nn.CrossEntropyLoss()

# Валидация модели

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Подсчет метрик
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    epoch_precision = precision_score(all_labels, all_preds, average='weighted')
    epoch_recall = recall_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall

print('\nTesting model...')

writer = SummaryWriter("my_logs/retrain")

test_loss, test_acc, test_f1, test_precision, test_recall = validate(
    model,
    test_loader,
    criterion,
    device
)

print(f'Test Loss: {test_loss:.10f}, Acc: {test_acc:.10f}, F1: {test_f1:.10f}, Precision: {test_precision:.10f}, Recall: {test_recall:.10f}')

# Логгирование метрик в tensorboard

writer.add_scalar('Loss/Test', test_loss, 0)
writer.add_scalar('Accuracy/Test', test_acc, 0)
writer.add_scalar('F1/Test', test_f1, 0)
writer.add_scalar('Precision/Test', test_precision, 0)
writer.add_scalar('Recall/Test', test_recall, 0)

writer.close()

print("Evaluation completed")
