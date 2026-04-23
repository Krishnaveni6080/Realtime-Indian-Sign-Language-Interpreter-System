# Indian Sign Language (ISL) Translation Project - Documentation Notes

## 1. Deep Learning Training Design (Why 100 Epochs?)

In the training script (`train_mediapipe1.py`), the variable is set to `EPOCHS = 100`. In machine learning, an "epoch" means the neural network has processed the entire dataset of 133,000+ gestures exactly one time.

Choosing **100 epochs** for the ISL network was a deliberate design choice based on four primary factors:

### A. The "Runway" Strategy (Model Checkpointing)
The training logic utilizes a checkpointing strategy. The limit of 100 epochs gives the model a long runway to train. Because the script only saves the model when the validation accuracy hits a new high, if the model reaches peak intelligence at an earlier epoch and then begins to overfit, it simply retains the best weights calculated at that peak. 100 is a safe ceiling to ensure maximum potential.

### B. Perfect Balance over Fast Training Times
The dataset uses MediaPipe 1D arrays (126 coordinate data points per frame) rather than highly demanding 3D raw image arrays. Because coordinate arrays are incredibly lightweight, each training epoch executes very quickly, providing maximum convergence over a highly reasonable time span without needing heavy computational hours.

### C. Complexity vs. Convergence (110 Classes)
The network must distinguish between **110 different signs**. The Adam optimizer gradually shifts weights to drop down the "slope" of mathematical loss. Detecting the subtle, complex differences between similar hand poses requires many iterative micro-adjustments for the optimizer to settle into the true minimum error.

### D. Industry Standards
In Deep Learning prototyping, 50 to 100 epochs is the universally accepted starting benchmark for Dense/Linear networks dealing with structured 1D numerical data, ensuring reliable data convergence.

---

## 2. Data Flow Diagram (DFD) Level 1 - Offline Training Phase

A **Data Flow Diagram (DFD) Level 1 for the Training Phase** breaks down the high-level process of how the system learns into specific, technical sub-processes. This helps visualize how raw data becomes an intelligent, trained algorithm.

### External Entities (Sources of Data)
*   **Signer / User (Open-Rectangle):** The human providing the physical sign language gestures.
*   **System Administrator / Developer (Open-Rectangle):** Provides training parameters (Learning Rate, Epochs, Batch Size).

### Main Processes (Circles)
These are the steps where data is actively transformed.

*   **Process 1.1: Image/Video Capture**
    *   *Input Data Flow:* Real-world gestures from the Signer.
    *   *Action:* OpenCV captures frames of the gesture via a webcam or reads pre-recorded video files.
    *   *Output Data Flow:* Raw image frames.

*   **Process 1.2: MediaPipe Landmark Extraction**
    *   *Input Data Flow:* Raw image frames pipeline.
    *   *Action:* The MediaPipe holistic/hands algorithm estimates 3D hand and body skeletal structures from the pixels.
    *   *Output Data Flow:* A 1D Array of 126 numerical coordinates (X, Y, Z).

*   **Process 1.3: Dataset Structuring**
    *   *Input Data Flow:* The extracted 126 coordinates and their True Class Label.
    *   *Action:* Formats the numerical arrays and appends them line-by-line along with their class mappings.
    *   *Output Data Flow:* Structured numerical data fed to a CSV Data Store.

*   **Process 1.4: Neural Network Training**
    *   *Input Data Flow:* Batches of coordinate arrays and labels.
    *   *Action:* The PyTorch script runs forward propagation, calculates Cross-Entropy loss, and uses the Adam optimizer to run backward propagation, updating weights over 100 Epochs.
    *   *Output Data Flow:* Trained mathematical weights and validation accuracies.

*   **Process 1.5: Best Model Checkpointing**
    *   *Input Data Flow:* Trained weights and epoch validation scores.
    *   *Action:* Compares validation accuracy with the previous high score; if better, it compiles and serializes the state dictionary.
    *   *Output Data Flow:* The finalized `.pth` binary file.

### Data Stores (Parallel Lines / Open-Ended Rectangles)
*   **`class_mapping_mediapipe11.json`:** Holds the string translation dictionary (e.g., `0 = "All_Gone"`).
*   **`landmarks_dataset11.csv`:** The centralized store of all extracted X,Y,Z coordinates.
*   **`best_model_mediapipe11.pth`:** The final serialized model weights, ready for live inference.
