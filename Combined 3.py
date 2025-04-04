import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
import time
import math
from collections import deque
from ultralytics import YOLO

# ==== Setup Paths ====
input_video_path = 'videos/8_pbks_kkr_sweep_3.mp4'
intermediate_output_path = 'output_videos/processed_video.mp4'
final_output_path = 'output_videos/final_cricket_analysis.mp4'
os.makedirs('output_videos', exist_ok=True)

# ==== Load Models ====
model_yolo = YOLO(os.path.join('runs','detect','train5','weights','best.pt'))
model_pose = tf.keras.models.load_model("cricket_shot_classifier.h5", compile=False)
model_pose.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ==== MediaPipe Pose ====
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# ==== Label Mapping ====
label_mapping = {
    0: "0. Cut Shot",
    1: "1. Cover Drive",
    2: "2. Straight Drive",
    3: "3. Pull Shot",
    4: "4. Leg Glance Shot",
    5: "5. Scoop Shot"
}

# ==== Helper Classes ====
class FixedSizeQueue:
    def __init__(self, max_size):
        self.queue = deque(maxlen=max_size)
    
    def add(self, item):
        self.queue.append(item)
    
    def pop(self):
        self.queue.popleft()
    
    def clear(self):
        self.queue.clear()

    def get_queue(self):
        return self.queue

    def __len__(self):  # ✅ This fixes the error
        return len(self.queue)

# ==== Helper Functions ====
def extract_keypoints(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
    else:
        keypoints = np.zeros(33 * 3)
    return keypoints

def angle_between_lines(m1, m2=1):
    if m1 != -1/m2:
        angle = math.degrees(math.atan(abs((m2 - m1) / (1 + m1 * m2))))
        return angle
    else:
        return 90.0

# ==== Step 1: Process Video and Save Intermediate Output ====
cap = cv2.VideoCapture(input_video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(intermediate_output_path, fourcc, fps, (frame_width, frame_height))

centroid_history = FixedSizeQueue(10)
angle = 0
start_time = time.time()
interval = 0.6
prev_frame_time = 0 
new_frame_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Pose model
    keypoints = extract_keypoints(frame)
    if not np.all(keypoints == 0):
        input_tensor = keypoints.reshape(1, -1)
        prediction = model_pose.predict(input_tensor, verbose=0)
        predicted_label = np.argmax(prediction)
        shot_name = label_mapping.get(predicted_label, "Unknown Shot")
    else:
        shot_name = "No Pose Detected"

    # YOLO tracking
    results = model_yolo.track(frame, persist=True, conf=0.35, verbose=False)
    boxes = results[0].boxes
    box = boxes.xyxy
    rows, cols = box.shape

    if len(box) != 0:
        for i in range(rows):
            x1, y1, x2, y2 = box[i]
            x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
            centroid_x = int((x1 + x2) / 2)
            centroid_y = int((y1 + y2) / 2)
            centroid_history.add((centroid_x, centroid_y))
            cv2.circle(frame, (centroid_x, centroid_y), 3, (0, 0, 255), -1)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    if len(centroid_history) > 1:
        for i in range(1, len(centroid_history)):
            cv2.line(frame, centroid_history.get_queue()[i-1], centroid_history.get_queue()[i], (255, 0, 0), 4)

        centroid_list = list(centroid_history.get_queue())
        x_diff = centroid_list[-1][0] - centroid_list[-2][0]
        y_diff = centroid_list[-1][1] - centroid_list[-2][1]

        if x_diff != 0:
            m1 = y_diff / x_diff
            if m1 == 1:
                angle = 90
            elif m1 != 0:
                angle = 90 - angle_between_lines(m1)
            if angle >= 45:
                print("Ball bounced")

        future_positions = [centroid_list[-1]]
        for i in range(1, 5):
            future_positions.append((centroid_list[-1][0] + x_diff * i, centroid_list[-1][1] + y_diff * i))

        for i in range(1, len(future_positions)):
            cv2.line(frame, future_positions[i-1], future_positions[i], (0, 255, 0), 4)
            cv2.circle(frame, future_positions[i], 3, (0, 0, 255), -1)

    # Overlay shot name & angle
    cv2.putText(frame, f"Shot: {shot_name}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.putText(frame, f"Angle: {angle:.2f} deg", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    out.write(frame)

cap.release()
out.release()

# ==== Step 2: Create Final Video Output ====
# -- Reopen video files
cap_original = cv2.VideoCapture(input_video_path)
cap_processed = cv2.VideoCapture(intermediate_output_path)

final_out = cv2.VideoWriter(final_output_path, fourcc, fps, (frame_width, frame_height))

# -- 1. Append Original Video
while True:
    ret, frame = cap_original.read()
    if not ret:
        break
    final_out.write(frame)
cap_original.release()

# -- 2. Add 2-Second "Cricket Analysis" Frame
title_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
cv2.putText(title_frame, "🏏 Cricket Analysis 🏏", (int(frame_width / 6), int(frame_height / 2)),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4, cv2.LINE_AA)
for _ in range(fps * 2):  # 2 seconds
    final_out.write(title_frame)

# -- 3. Append Processed Video in Slow Motion (write each frame twice)
while True:
    ret, frame = cap_processed.read()
    if not ret:
        break
    final_out.write(frame)
    final_out.write(frame)
    final_out.write(frame)
    final_out.write(frame)

cap_processed.release()
final_out.release()

print(f"✅ Final cricket analysis video saved at: {final_output_path}")
