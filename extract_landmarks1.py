import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["GLOG_minloglevel"] = "2" # Suppress MediaPipe C++ Logs
import cv2
import csv
import json
import gc
import mediapipe as mp
import mediapipe as mp

DATASET_PATH = r"C:\Users\krish\Downloads\ISL_FINAL\images"
OUTPUT_CSV = "landmarks_dataset11.csv"
MAPPING_SAVE_PATH = "class_mapping_mediapipe11.json"

# Initialize MediaPipe Hands utility
mp_hands = mp.solutions.hands

def process_dataset():
    # Instantiate inside function to manage its lifecycle
    hands = mp_hands.Hands(
        static_image_mode=True, 
        max_num_hands=2, 
        min_detection_confidence=0.5
    )
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    classes = sorted(os.listdir(DATASET_PATH))
    print(f"Found {len(classes)} classes.")

    # Save class mapping
    with open(MAPPING_SAVE_PATH, 'w') as f:
        json.dump({i: c for i, c in enumerate(classes)}, f)

    total_images = 0
    valid_hands = 0

    with open(OUTPUT_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        
        # Write CSV Header: label, l_x0...l_z20, r_x0...r_z20 (126 features)
        header = ['label']
        for i in range(21):
            header.extend([f'l_x{i}', f'l_y{i}', f'l_z{i}'])
        for i in range(21):
            header.extend([f'r_x{i}', f'r_y{i}', f'r_z{i}'])
        writer.writerow(header)

        for label_idx, class_name in enumerate(classes):
            class_dir = os.path.join(DATASET_PATH, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            print(f"  -> Processing class '{class_name}' [{label_idx + 1}/{len(classes)}]")
            images = os.listdir(class_dir)
            
            for img_name in images:
                total_images += 1
                img_path = os.path.join(class_dir, img_name)
                
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                # MediaPipe requires RGB format
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                
                if results.multi_hand_landmarks:
                    valid_hands += 1
                    
                    left_hand_data = [0.0] * 63
                    right_hand_data = [0.0] * 63
                    
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        hand_label = handedness.classification[0].label
                        
                        wrist = hand_landmarks.landmark[0]
                        base_x, base_y, base_z = wrist.x, wrist.y, wrist.z
                        
                        # Find the maximum distance from the wrist (hand bounding radius)
                        # This enables perfect scale/distance invariance! 
                        max_dist = 1e-6
                        for lm in hand_landmarks.landmark:
                            dist = ((lm.x - base_x)**2 + (lm.y - base_y)**2 + (lm.z - base_z)**2)**0.5
                            if dist > max_dist:
                                max_dist = dist
                                
                        hand_data = []
                        for lm in hand_landmarks.landmark:
                            # Normalize all coordinates precisely to the hand's radius
                            hand_data.extend([
                                (lm.x - base_x) / max_dist, 
                                (lm.y - base_y) / max_dist, 
                                (lm.z - base_z) / max_dist
                            ])
                            
                        if hand_label == 'Left':
                            left_hand_data = hand_data
                        else:
                            right_hand_data = hand_data
                            
                    row = [label_idx] + left_hand_data + right_hand_data
                    writer.writerow(row)
                    
                # Explicitly clear memory each frame to prevent OOM
                del image
                del image_rgb
                del results
                
                if total_images % 100 == 0:
                    gc.collect()
                    
                # Restart MediaPipe periodically to mitigate internal C++ memory leaks
                if total_images % 500 == 0:
                    hands.close()
                    hands = mp_hands.Hands(
                        static_image_mode=True, 
                        max_num_hands=2, 
                        min_detection_confidence=0.5
                    )

    # Close the MediaPipe object finally
    hands.close()

    print(f"\n=====================================")
    print(f"        EXTRACTION COMPLETE          ")
    print(f"=====================================")
    print(f"Total processed images by CV2       : {total_images}")
    print(f"Successfully extracted hand vectors : {valid_hands}")
    print(f"Dropped images (no hand found)      : {total_images - valid_hands}")
    print(f"Saved dataset ready for ML training : {OUTPUT_CSV}")

if __name__ == "__main__":
    process_dataset()
