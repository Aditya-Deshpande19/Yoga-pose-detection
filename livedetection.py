import cv2
import mediapipe as mp
import numpy as np
import torch
import pickle

# === MODEL CLASS (same as training) ===
class PoseClassifier(torch.nn.Module):
    def __init__(self, input_dim=132, hidden1=256, hidden2=128, output_dim=10):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden1),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden1, hidden2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden2, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# === DEVICE ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD LABEL ENCODER ===
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

num_classes = len(le.classes_)

# === LOAD MODEL ===
model = PoseClassifier(output_dim=num_classes)
model.load_state_dict(torch.load("pose_classifier.pth", map_location=device))
model.to(device)
model.eval()

# === MEDIAPIPE SETUP ===
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# === START WEBCAM ===
cap = cv2.VideoCapture(0)

print("✅ Press 'q' to exit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process pose
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # === EXTRACT KEYPOINTS ===
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])

        keypoints = np.array(keypoints, dtype=np.float32).reshape(1, -1)

        # === PREDICTION ===
        input_tensor = torch.tensor(keypoints).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            pred_idx = output.argmax(dim=1).item()
            pred_label = le.inverse_transform([pred_idx])[0]

        # === DRAW LANDMARKS ===
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

        # === SHOW PREDICTION ===
        cv2.putText(
            frame,
            f"Pose: {pred_label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    else:
        cv2.putText(
            frame,
            "No Pose Detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )

    # === DISPLAY ===
    cv2.imshow("AI Yoga Trainer", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()