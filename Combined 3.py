import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
import time
from collections import deque
from ultralytics import YOLO

# Load the Cricket Shot Classification Model
model = tf.keras.models.load_model("cricket_shot_classifier.h5", compile=False)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Label Mapping for Shots
label_mapping = {
    0: "Cut Shot",
    1: "Cover Drive",
    2: "Straight Drive",
    3: "Pull Shot",
    4: "Leg Glance",
    5: "Scoop Shot"
}

# Load the Ball Tracking YOLO Model
yolo_model = YOLO(os.path.join('runs', 'detect', 'train5', 'weights', 'best.pt'))

# Video Input & Output
input_video = "videos/8_pbks_kkr_sweep_3.mp4"
output_video = "final_processed_cricket.mp4"

# Open Video
cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("Error: Could not open input video.")
    exit()

# Get Video Properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define Video Writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# Queue for ball tracking
class FixedSizeQueue:
    def __init__(self, max_size):
        self.queue = deque(maxlen=max_size)

    def add(self, item):
        self.queue.append(item)

    def get_queue(self):
        return self.queue

centroid_history = FixedSizeQueue(10)

# --- 1. Write the Original Video to the Output ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)

cap.release()

# --- 2. Add Stylish "Cricket Analysis" Screen ---
analysis_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
cv2.putText(analysis_frame, " CRICKET ANALYSIS ", 
            (int(frame_width * 0.15), int(frame_height / 2)), 
            cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 5, cv2.LINE_AA)

# Add a glow effect
for _ in range(fps * 2):  # 2 seconds
    out.write(analysis_frame)

# --- 3. Process Video with 3X Slow Motion, Shot Classification & Ball Tracking ---
cap = cv2.VideoCapture(input_video)

# Create Shot Label Banner (All Possible Shots)
banner = np.zeros((100, frame_width, 3), dtype=np.uint8)
for idx, shot_name in label_mapping.items():
    x_pos = int((idx + 0.5) * (frame_width / len(label_mapping)))
    cv2.putText(banner, shot_name, (x_pos - 50, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Slow Motion Effect (Duplicate Frames 3 Times)
    for _ in range(3):  
        processed_frame = frame.copy()

        # Extract Pose Keypoints
        keypoints = np.zeros(33 * 3)  # Default if no pose detected
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if results.pose_landmarks:
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
        
        # Predict Shot
        keypoints = keypoints.reshape(1, -1)
        prediction = model.predict(keypoints)
        predicted_label = np.argmax(prediction)
        detected_shot = label_mapping.get(predicted_label, "Unknown Shot")

        # Track the Ball with YOLO
        results = yolo_model.track(frame, persist=True, conf=0.35, verbose=False)
        boxes = results[0].boxes
        box = boxes.xyxy
        if len(box) != 0:
            for i in range(box.shape[0]):
                x1, y1, x2, y2 = map(int, box[i])
                centroid_x, centroid_y = (x1 + x2) // 2, (y1 + y2) // 2
                centroid_history.add((centroid_x, centroid_y))

        # Draw Ball Trajectory with a Thicker, Translucent Effect
        for i in range(1, len(centroid_history.get_queue())):
            cv2.line(processed_frame, centroid_history.get_queue()[i-1], 
                     centroid_history.get_queue()[i], (0, 255, 255, 100), 12)  # Wider, More Transparent

        # Highlight Detected Shot in Oval Balloon **(Now Covering Text & Translucent)**
        overlay = banner.copy()
        detected_x = int((predicted_label + 0.5) * (frame_width / len(label_mapping)))

        # Translucent Bubble Effect
        bubble_color = (0, 0, 255)  # Red Bubble
        cv2.ellipse(overlay, (detected_x, 60), (80, 40), 0, 0, 360, bubble_color, -1)  # Full Coverage of Text
        cv2.addWeighted(overlay, 0.5, banner, 0.5, 0, banner)  # Make Bubble Translucent

        # Overlay Detected Shot Text
        cv2.putText(banner, detected_shot, (detected_x - 50, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Blend the Banner with Highlight
        alpha = 0.6
        processed_frame[:100, :] = cv2.addWeighted(processed_frame[:100, :], 1 - alpha, banner, alpha, 0)

        # Overlay Detected Shot Text on Frame
        cv2.putText(processed_frame, f"Detected Shot: {detected_shot}", 
                    (30, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        out.write(processed_frame)

cap.release()
out.release()

print(f"✅ Final Processed Video Saved: {output_video}")
