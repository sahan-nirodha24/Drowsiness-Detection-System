# Driver Drowsiness Detection System

This project implements a real-time Driver Drowsiness Detection System using Computer Vision and Machine Learning. It provides two distinct approaches to detect drowsiness:
1.  **CNN Model:** A Deep Learning approach using a custom MobileNetV2 architecture.
2.  **SVM Model:** A Machine Learning approach using HOG (Histogram of Oriented Gradients) and LBP (Local Binary Patterns) features with a Support Vector Machine classifier.

Both implementations use a webcam to monitor the driver's eyes and face, triggering audio and visual alerts when drowsiness is detected.

## 📂 Project Structure

The project is organized into two main directories, each containing a standalone implementation:

```
driver-drowsiness-detection/
│
├── 📁 CNN Model/               # Deep Learning Implementation
│   ├── drowsiness_app.py       # Main application script for CNN
│   ├── Drowsiness_Detection_CNN.ipynb  # Notebook for training the CNN model
│   ├── split.py                # Script for dataset splitting
│   ├── 📁 Models/              # Saved .h5 models
│   ├── 📁 Imageset/            # Raw dataset
│   └── 📁 Processed_Dataset/   # Train/Test/Val split
│
└── 📁 SVM Model/               # Machine Learning Implementation
    ├── drowsiness_app.py       # Main application script for SVM
    ├── Drowsiness_Detection_SVM.ipynb  # Notebook for training the SVM model
    ├── split_dataset.py        # Script for dataset splitting
    ├── svm_model.pkl           # Trained SVM model
    ├── hog_parameters.pkl      # Saved feature extraction parameters
    └── label_encoder.pkl       # Saved label encoder
```

## 🛠 Prerequisites

Ensure you have Python installed (Python 3.13 or higher recommended). Install the required dependencies using pip:

```bash
pip install opencv-python numpy tensorflow scikit-learn scikit-image pyttsx3 pandas matplotlib
```

*Note: `winsound` is used for beep alerts and is included by default on Windows.*

## 🚀 Usage

### 1. Running the CNN Model
The CNN model uses Deep Learning for robust detection.

1.  Navigate to the `CNN Model` directory:
    ```bash
    cd "CNN Model"
    ```
2.  Run the application:
    ```bash
    python drowsiness_app.py
    ```
    - **Controls:**
        - `q`: Quit the application.
        - `i`: Invert detection logic (if needed).

### 2. Running the SVM Model
The SVM model uses classical computer vision features (HOG + LBP) for detection.

1.  Navigate to the `SVM Model` directory:
    ```bash
    cd "SVM Model"
    ```
2.  Run the application:
    ```bash
    python drowsiness_app.py
    ```
    - **Controls:**
        - `q`: Quit the application.

## 🎯 Features

*   **Real-time Detection:** Uses your webcam to monitor face and eyes.
*   **Hybrid Logic:** Combines geometric eye-closure detection (using Haar Cascades) with model-based drowsiness classification.
*   **Audio Alerts:**
    *   **Beep:** Instant alert noise.
    *   **Voice:** "Wake up! You are drowsy" voice command using `pyttsx3`.
*   **Visual Feedback:**
    *   **Green/Red Boxes:** Highlights face and eyes based on status.
    *   **Status Text:** Displays "Active", "DROWSY", or "EYES CLOSED!".
    *   **Probability Bar:** (SVM Only) Shows the calculated probability of drowsiness.

## 📊 Training Your Own Models
If you wish to retrain the models with your own dataset:

1.  Place your images in the `Imageset` folder within the respective model directory.
2.  Run the split script (`split.py` or `split_dataset.py`) to organize data.
3.  Open the corresponding Jupyter Notebook (`.ipynb`) and run the cells to train and save the new model.
