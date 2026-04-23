import os
import csv
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

CSV_PATH = "landmarks_dataset11.csv"
MAPPING_PATH = "class_mapping_mediapipe11.json"
MODEL_SAVE_PATH = "best_model_mediapipe11.pth"
BATCH_SIZE = 32
EPOCHS = 100

class LandmarkDataset(Dataset):
    def __init__(self, csv_file):
        self.data = []
        self.labels = []
        
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader) # Skip header
            for row in reader:
                if not row: continue
                self.labels.append(int(row[0]))
                self.data.append([float(x) for x in row[1:]])
                
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class LandmarkNN(nn.Module):
    def __init__(self, input_size=126, num_classes=10):
        super(LandmarkNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Dataset {CSV_PATH} not found!")
        return
        
    with open(MAPPING_PATH, 'r') as f:
        class_mapping = json.load(f)
        num_classes = len(class_mapping)
        
    dataset = LandmarkDataset(CSV_PATH)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LandmarkNN(input_size=126, num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_acc = 0.0
    print("Starting Training on Coordinate Dataset...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / train_size
        
        # Validation
        model.eval()
        correct = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                
        val_acc = 100 * correct / val_size
        v_loss = val_loss / val_size
        
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {epoch_loss:.4f} | Val Loss: {v_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            
    print(f"Training Complete! Best Accuracy: {best_acc:.2f}%")
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()
