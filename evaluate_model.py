


import torch
import json
from train_mediapipe1 import LandmarkDataset, LandmarkNN
import os

CSV_PATH = "landmarks_dataset11.csv"
MAPPING_PATH = "class_mapping_mediapipe11.json"
MODEL_SAVE_PATH = "best_model_mediapipe11.pth"

with open(MAPPING_PATH, 'r') as f:
    class_mapping = json.load(f)
    num_classes = len(class_mapping)

dataset = LandmarkDataset(CSV_PATH)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LandmarkNN(input_size=126, num_classes=num_classes).to(device)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Overall Accuracy: {100 * correct / total:.2f}%")
