# 🤟 Realtime Indian Sign Language Interpreter System

A real-time Indian Sign Language recognition system powered by **MediaPipe**, **PyTorch**, and **Flask**. The system uses hand landmark extraction and a lightweight neural network for **isolated, word-level classification** of ISL signs and converts them to speech — bridging communication for the hearing impaired.

---



## ✨ Features

- 🖐️ **Real-time hand landmark extraction** using Google MediaPipe
- 🧠 **Neural network classifier** with ~95%+ overall accuracy across 110+ ISL signs
- 🔊 **Text-to-speech output** for detected signs (Windows Speech Synthesis)
- 🌐 **Flask web interface** — accessible from any device on the local network
- 📊 **Confidence scoring** with a prediction buffer for stable, noise-free results
- 🤝 **Two-hand support** — handles both left and right hand landmarks simultaneously

---

## 🏗️ System Architecture

```
Webcam Input
    │
    ▼
MediaPipe Hands  ──►  Landmark Extraction (21 keypoints × 2 hands = 126 features)
    │
    ▼
LandmarkNN (PyTorch)  ──►  Sign Classification (110+ classes)
    │
    ▼
Prediction Buffer  ──►  Stable Sign Detection
    │
    ├──►  Flask Web UI  (Live video + confidence display)
    └──►  Text-to-Speech  (Windows Speech API)
```

---

## 📁 Project Structure

```
Final_ISL/
├── app.py                        # Flask web server & routes
├── engine.py                     # Core ISL engine (camera, model, TTS)
├── train_mediapipe1.py           # Model training script
├── extract_landmarks1.py         # Landmark extraction & dataset builder
├── evaluate_model.py             # Model evaluation script
├── performance_eval.py           # Detailed performance metrics
├── best_model_mediapipe11.pth    # Trained PyTorch model weights
├── class_mapping_mediapipe11.json # Class index → sign label mapping
├── confusion_matrix.png          # Confusion matrix from evaluation
├── performance_output.txt        # Full classification report
├── templates/
│   └── index.html                # Web UI template
├── static/
│   ├── css/                      # Stylesheets
│   └── js/                       # Frontend scripts
├── requirements.txt
└── .gitignore
```

> **Note:** The training dataset (`landmarks_dataset11.csv`, ~214MB) and raw image dataset (`images/`) are excluded from this repository due to size. You can regenerate the dataset using `extract_landmarks1.py`.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Webcam
- Windows OS (for TTS via Windows Speech API)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Krishnaveni6080/Final_ISL.git
cd Final_ISL

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
python app.py
```

Then open your browser and go to: **[http://localhost:5000](http://localhost:5000)**

---

## 🧠 Model Details

| Property         | Value                         |
|-----------------|-------------------------------|
| Architecture     | Fully Connected Neural Network |
| Input Features   | 126 (21 landmarks × 3 coords × 2 hands) |
| Hidden Layers    | 256 → 128 → 64                |
| Regularization   | BatchNorm + Dropout (0.2–0.3) |
| Output Classes   | 110+                          |
| Optimizer        | Adam (lr=0.001)               |
| Loss Function    | Cross-Entropy Loss            |
| Training Epochs  | 100                           |
| Best Val Accuracy| ~95%+                         |

---

## 📊 Performance

The model was evaluated on a held-out test set. Key metrics:

- **Overall Accuracy**: ~95%+
- Common alphabets (A–Z) and digits (0–9): **>97% F1-score**
- Full per-class results: see [`performance_output.txt`](performance_output.txt)
- Confusion matrix: see [`confusion_matrix.png`](confusion_matrix.png)

---

## 🔄 Retraining the Model

If you want to retrain with your own data:

```bash
# Step 1: Collect images and extract landmarks
python extract_landmarks1.py

# Step 2: Train the model
python train_mediapipe1.py

# Step 3: Evaluate
python evaluate_model.py
```

---

## 🛠️ Tech Stack

| Component      | Technology                  |
|---------------|------------------------------|
| Hand Tracking  | MediaPipe Hands              |
| ML Framework   | PyTorch                      |
| Web Framework  | Flask                        |
| Video Capture  | OpenCV                       |
| Text-to-Speech | Windows Speech Synthesis API |
| Frontend       | HTML, CSS, JavaScript        |

---


## 📄 License

This project is for academic and educational purposes.
