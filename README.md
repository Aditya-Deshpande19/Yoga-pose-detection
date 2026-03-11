Yoga Pose Detection using MediaPipe and Machine Learning

This project implements a Yoga Pose Detection system using MediaPipe for keypoint extraction and machine learning models for pose classification. It allows automatic recognition of yoga poses from images, which can be used for fitness apps, personal training, or posture correction systems.

Key Features:

Pose Keypoint Extraction:

Uses MediaPipe Pose to detect 33 body landmarks (keypoints) from images.

Each landmark includes x, y, z coordinates and visibility.

Missing keypoints are padded with zeros to maintain a consistent input shape (132 features per image).

Dataset Handling:

Supports combining old training data (previously saved features and labels) with new images for incremental training.

Automatically handles mismatches between feature and label counts.

Machine Learning Models:

RandomForestClassifier (scikit-learn) for fast and robust pose classification.

Optionally, can use TensorFlow/Keras neural networks for advanced training and export to TFLite for mobile deployment.

Label Encoding:

Converts pose names into numeric labels using LabelEncoder.

Model Saving:

RandomForest models are saved using pickle or joblib.

Neural network models can be saved as .h5 (Keras) and converted to .tflite for lightweight deployment.

Extensible & Flexible:

Easy to add new yoga poses by adding images to the dataset folder.

Automatically extracts keypoints, encodes labels, and retrains the model.

Technology Stack / Libraries Used:

Python 3.x

MediaPipe – for pose landmark detection

OpenCV – for image reading and preprocessing

NumPy – for numerical operations

scikit-learn – for RandomForestClassifier and label encoding

pickle / joblib – for saving and loading trained models

TensorFlow/Keras (optional) – for neural network models and TFLite conversion

Use Case:

Fitness apps for yoga training and pose correction

Real-time posture monitoring using camera input

Incremental learning with new poses for custom datasets
