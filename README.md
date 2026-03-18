# Yoga Pose Detection using MediaPipe and Machine Learning

## Project Overview
This project implements a **Yoga Pose Detection System** using **MediaPipe for keypoint extraction** and **Machine Learning models for pose classification**.

The system automatically recognizes yoga poses from images, which can be used for **fitness applications, posture correction systems, and personal training platforms**.

---

## Key Features

### Pose Keypoint Extraction
- Uses **MediaPipe Pose** to detect **33 body landmarks (keypoints)** from images.
- Each landmark includes:
  - x coordinate
  - y coordinate
  - z coordinate
  - visibility score
- Missing keypoints are **padded with zeros** to maintain a consistent input shape.

Total features per image:

132 features

---

### Dataset Handling
- Supports **combining previously saved training data with new images**.
- Enables **incremental training**.
- Automatically handles mismatches between:
  - feature counts
  - label counts

---

### Machine Learning Models

#### Random Forest Classifier
- Implemented using **scikit-learn**
- Fast training and reliable classification
- Works well for structured keypoint data

#### Neural Network Model (Optional)
- Implemented using **TensorFlow / Keras**
- Suitable for advanced training
- Can be exported to **TensorFlow Lite (TFLite)** for mobile deployment

---

### Label Encoding
Pose names are converted into **numeric labels** using **LabelEncoder**.

Example:

Tree Pose → 0  
Warrior Pose → 1  
Downward Dog → 2

---

### Model Saving

#### Random Forest Model
Saved using:
- pickle
- joblib

#### Neural Network Model
Saved as:
- .h5 (Keras model)
- .tflite (TensorFlow Lite model)

---

### Extensible and Flexible System
New yoga poses can be added easily.

Steps:
1. Add images of the new pose into the dataset folder
2. Run the training script
3. The system will automatically:
   - extract keypoints
   - encode labels
   - retrain the model

---

## Technology Stack

| Technology | Purpose |
|-----------|--------|
| Python 3.x | Core programming language |
| MediaPipe | Pose landmark detection |
| OpenCV | Image processing |
| NumPy | Numerical operations |
| scikit-learn | Machine learning models |
| pickle / joblib | Model saving |
| TensorFlow / Keras | Neural network training |
| TensorFlow Lite | Mobile deployment |

---

## Use Cases

### Fitness Applications
Used in yoga training platforms to automatically recognize poses.

### Real-Time Posture Monitoring
Can be integrated with webcam input for posture analysis.

### Incremental Learning Systems
Allows continuous improvement by adding new poses to the dataset.

### Mobile Fitness Applications
Using TensorFlow Lite, the model can run on smartphones or embedded devices.

---

## Future Improvements
- Real-time webcam-based pose detection
- Pose accuracy scoring
- Mobile application integration
- Deep learning models for improved accuracy
- Real-time posture correction feedback system

---

## Author

Aditya Deshpande

GitHub Profile  
https://github.com/Aditya-Deshpande19

Author

Aditya Deshpande
