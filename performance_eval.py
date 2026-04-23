import torch
import json
import numpy as np
from train_mediapipe1 import LandmarkDataset, LandmarkNN
import os

try:
    from sklearn.metrics import confusion_matrix, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

CSV_PATH = "landmarks_dataset11.csv"
MAPPING_PATH = "class_mapping_mediapipe11.json"
MODEL_SAVE_PATH = "best_model_mediapipe11.pth"

def evaluate_performance():
    if not os.path.exists(CSV_PATH) or not os.path.exists(MAPPING_PATH) or not os.path.exists(MODEL_SAVE_PATH):
        print("Missing required files (dataset, mapping, or model).")
        return

    with open(MAPPING_PATH, 'r') as f:
        class_mapping = json.load(f)
        num_classes = len(class_mapping)
        
    # Extract class names in order
    class_names = [class_mapping[str(i)] for i in range(num_classes)]
    
    dataset = LandmarkDataset(CSV_PATH)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LandmarkNN(input_size=126, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = np.mean(all_preds == all_labels) * 100
    
    print(f"Overall Accuracy: {accuracy:.2f}%\n")
    
    if SKLEARN_AVAILABLE:
        print("Classification Report:")
        print(classification_report(all_labels, all_preds, labels=np.arange(num_classes), target_names=class_names))
        
        cm = confusion_matrix(all_labels, all_preds, labels=np.arange(num_classes))
        print("\nConfusion Matrix:")
        print(cm)
        
        if PLOT_AVAILABLE:
            plt.figure(figsize=(40, 40))
            sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix', fontsize=30)
            plt.ylabel('True Label', fontsize=20)
            plt.xlabel('Predicted Label', fontsize=20)
            plt.xticks(rotation=90, fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            plt.tight_layout()
            plt.savefig('confusion_matrix.png', dpi=150)
            print("\nSaved confusion matrix plot as 'confusion_matrix.png'")
    else:
        print("scikit-learn is not available. Please run 'pip install scikit-learn' to see the full classification report and confusion matrix.")

if __name__ == '__main__':
    evaluate_performance()
