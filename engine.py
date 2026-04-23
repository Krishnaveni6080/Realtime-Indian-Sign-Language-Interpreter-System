import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["GLOG_minloglevel"] = "2" # Suppress MediaPipe C++ Logs
import cv2
import json
import torch
import threading
import queue
import numpy as np
import time
from collections import deque
import mediapipe as mp
import torch.nn as nn
import subprocess

MODEL_PATH = "best_model_mediapipe11.pth"
MAPPING_PATH = "class_mapping_mediapipe11.json"
CONFIDENCE_THRESHOLD = 0.55
BUFFER_SIZE = 15
COOLDOWN_SECONDS = 3.0

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

class ISLEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_mapping = {}
        self.tts_queue = queue.Queue()
        self.prediction_buffer = deque(maxlen=BUFFER_SIZE)
        
        self.last_spoken_word = None
        self.last_spoken_time = 0
        
        self.current_sign = "---"
        self.current_confidence = 0.0
        self.sentence_history = []
        
        self.is_running = True
        self.camera_active = False
        self.cap = None
        
        # Precompute blank frame for offline state
        blank = np.zeros((480, 640, 3), np.uint8)
        ret, jpeg = cv2.imencode('.jpg', blank)
        self.blank_frame = jpeg.tobytes()

        threading.Thread(target=self.tts_worker, daemon=True).start()
        self.initialize_engine()

    def tts_worker(self):
        while self.is_running:
            try:
                text = self.tts_queue.get(timeout=1.0)
                try:
                    ps_script = f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')"
                    subprocess.Popen(['powershell', '-Command', ps_script], 
                                     creationflags=subprocess.CREATE_NO_WINDOW)
                except Exception as eval_e:
                    print(f"TTS Warning: {eval_e}")
                    
                self.tts_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                print(f"TTS Queue Error: {e}")

    def speak_text(self, text):
        self.tts_queue.put(text)

    def initialize_engine(self):
        try:
            with open(MAPPING_PATH, 'r') as f:
                class_mapping = json.load(f)
                self.class_mapping = {int(k): v for k, v in class_mapping.items()}
        except FileNotFoundError:
            print(f"Error: {MAPPING_PATH} not found.")
            return

        num_classes = len(self.class_mapping)
        self.model = LandmarkNN(input_size=126, num_classes=num_classes)
        
        try:
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self.model = self.model.to(self.device)
            if self.device.type == "cuda":
                self.model = self.model.half()
            self.model.eval()
        except FileNotFoundError:
            print(f"Error: {MODEL_PATH} not found.")
            return

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def start_camera(self):
        if not self.camera_active:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if self.cap.isOpened():
                self.camera_active = True
                self.current_sign = "Waiting..."
                self.current_confidence = 0.0
                return True
        return False
        
    def stop_camera(self):
        self.camera_active = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.current_sign = "---"
        self.current_confidence = 0.0
        return True

    def get_frame(self):
        if not self.camera_active or not self.cap or not self.cap.isOpened():
            time.sleep(0.1) # Prevent CPU spinning
            return self.blank_frame

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return self.blank_frame

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        image_rgb.flags.writeable = False
        results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True
        
        if results.multi_hand_landmarks and self.model:
            left_hand_data = [0.0] * 63
            right_hand_data = [0.0] * 63
            
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                wrist = hand_landmarks.landmark[0]
                base_x, base_y, base_z = wrist.x, wrist.y, wrist.z
                
                max_dist = 1e-6
                for lm in hand_landmarks.landmark:
                    dist = ((lm.x - base_x)**2 + (lm.y - base_y)**2 + (lm.z - base_z)**2)**0.5
                    if dist > max_dist:
                        max_dist = dist
                        
                hand_data = []
                for lm in hand_landmarks.landmark:
                    hand_data.extend([
                        (lm.x - base_x) / max_dist, 
                        (lm.y - base_y) / max_dist, 
                        (lm.z - base_z) / max_dist
                    ])
                    
                if hand_label == 'Left':
                    left_hand_data = hand_data
                else:
                    right_hand_data = hand_data
                    
            row = left_hand_data + right_hand_data
            input_tensor = torch.tensor(row, dtype=torch.float32).unsqueeze(0).to(self.device)
            if self.device.type == "cuda":
                input_tensor = input_tensor.half()

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)
                
                self.prediction_buffer.append((predicted.item(), confidence.item()))

            if len(self.prediction_buffer) == BUFFER_SIZE:
                classes = [p[0] for p in self.prediction_buffer]
                avg_conf = sum(p[1] for p in self.prediction_buffer) / BUFFER_SIZE
                most_common_class = max(set(classes), key=classes.count)
                class_name = self.class_mapping.get(most_common_class, "---")
                
                is_stable = classes.count(most_common_class) >= (BUFFER_SIZE * 0.5)
                
                if is_stable and avg_conf > CONFIDENCE_THRESHOLD:
                    self.current_sign = class_name
                    self.current_confidence = avg_conf

                    current_time = time.time()
                    if class_name != self.last_spoken_word or (current_time - self.last_spoken_time) > COOLDOWN_SECONDS:
                        self.sentence_history.append(class_name)
                        if len(self.sentence_history) > 50:
                            self.sentence_history.pop(0)
                        
                        self.speak_text(class_name)
                        self.last_spoken_word = class_name
                        self.last_spoken_time = current_time
                else:
                    self.current_confidence = avg_conf
                    self.current_sign = "Waiting..."

        frame = cv2.flip(frame, 1)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            return self.blank_frame
        return jpeg.tobytes()

    def get_stats(self):
        return {
            "sign": self.current_sign,
            "confidence": self.current_confidence,
            "history": " ".join(self.sentence_history[-15:]),
            "camera_active": self.camera_active
        }

    def release(self):
        self.is_running = False
        self.stop_camera()
