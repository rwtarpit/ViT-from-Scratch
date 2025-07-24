import torch.nn as nn
import torch
from ViT import build_ViT
import torch.optim as optim
from data_and_preprocessing import trainloader, testloader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import amp
from tqdm import tqdm

for_cifar={
  "img_size": 32,
  "patch_size": 4,
  "embedding_dim": 192,
  "heads": 3,
  "layers": 6,
  "mlp_dim": 384,
  "dropout": 0.1
    }


model = build_ViT(32,32,3,4,192,384,6,10,0.1,3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
epochs = 100
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    loop = tqdm(dataloader, desc="Training", leave=False)

    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)  
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        loop.set_postfix(loss=running_loss / total, acc=100. * correct / total)

    return running_loss / total, 100. * correct / total

def evaluate(model, dataloader, criterion):
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100. * correct / total


for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer)
    val_loss, val_acc = evaluate(model, testloader, criterion)
    scheduler.step()
    print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
    print(f"Val   Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
   
