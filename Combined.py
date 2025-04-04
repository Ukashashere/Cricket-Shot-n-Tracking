import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
import math
import time
from collections import deque
from ultralytics import YOLO

# ==================== Load Models ====================

# Load Cricket Shot Classification Model
cricket_model = tf.keras.models.load_model("cricket_shot_classifier.h5", compile=False)
cricket_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load YOLO Model for object tracking
yolo_model = YOLO(os.path.join('runs', 'detect', 'train5', 'weights', 'best.pt'))

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Label Mapping for Cricket Shot Classification
label_mapping = {
    0: "0. Cut Shot",
    1: "1. Cover Drive",
    2: "2. Straight Drive",
    3: "3. Pull Shot",
    4: "4. Leg Glance Shot",
    5: "5. Scoop Shot"
}

print("✅ Models Loaded Successfully!")


# ==================== Function Definitions ====================

def extract_keypoints(image):
    """Extract keypoints from image using MediaPipe Pose."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
    else:
        keypoints = np.zeros(33 * 3)  # 33 landmarks, each with x, y, z

    return keypoints


class FixedSizeQueue:
    """A queue to store object centroids for tracking."""
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

    def __len__(self):
        return len(self.queue)


def angle_between_lines(m1, m2=1):
    """Calculate angle between two lines using slopes."""
    if m1 != -1 / m2:
        angle = math.degrees(math.atan(abs((m2 - m1) / (1 + m1 * m2))))
        return angle
    return 90.0


def process_video(input_video_path, output_video_path):
    """Processes video, classifies cricket shots, tracks ball movement, and overlays results."""
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open input video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define codec and create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Ball Tracking Variables
    centroid_history = FixedSizeQueue(10)
    start_time = time.time()
    interval = 0.6
    prev_frame_time = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        new_frame_time = time.time()
        fps_text = int(1 / (new_frame_time - prev_frame_time))
        prev_frame_time = new_frame_time

        current_time = time.time()
        if current_time - start_time >= interval and len(centroid_history) > 0:
            centroid_history.pop()
            start_time = current_time

        # ========== Step 1: Cricket Shot Classification ==========
        keypoints = extract_keypoints(frame)
        if np.all(keypoints == 0):
            shot_name = "No Pose Detected"
        else:
            keypoints = keypoints.reshape(1, -1)
            prediction = cricket_model.predict(keypoints)
            predicted_label = np.argmax(prediction)
            shot_name = label_mapping.get(predicted_label, "Unknown Shot")

        # ========== Step 2: Ball Tracking with YOLO ==========
        results = yolo_model.track(frame, persist=True, conf=0.35, verbose=False)
        boxes = results[0].boxes
        box = boxes.xyxy
        rows, cols = box.shape

        if len(box) != 0:
            for i in range(rows):
                x1, y1, x2, y2 = box[i]
                x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()

                # Calculate Centroid
                centroid_x = int((x1 + x2) / 2)
                centroid_y = int((y1 + y2) / 2)
                centroid_history.add((centroid_x, centroid_y))

                # Draw Bounding Box & Centroid
                cv2.circle(frame, (centroid_x, centroid_y), radius=3, color=(0, 0, 255), thickness=-1)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        # Draw Trajectory
        if len(centroid_history) > 1:
            for i in range(1, len(centroid_history)):
                cv2.line(frame, centroid_history.get_queue()[i-1], centroid_history.get_queue()[i], (255, 0, 0), 4)

        # Bounce Detection
        angle = 0
        if len(centroid_history) > 1:
            centroid_list = list(centroid_history.get_queue())
            x_diff = centroid_list[-1][0] - centroid_list[-2][0]
            y_diff = centroid_list[-1][1] - centroid_list[-2][1]

            if x_diff != 0:
                m1 = y_diff / x_diff
                angle = 90 - angle_between_lines(m1)
                if angle >= 45:
                    print("Ball bounced")

        # ========== Overlay Information on Video ==========
        cv2.putText(frame, f"Shot: {shot_name}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.putText(frame, f"Angle: {angle:.2f} degrees", (20, 80), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"FPS: {fps_text}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Save Frame to Output Video
        out.write(frame)

        # Display Video Frame
        frame_resized = cv2.resize(frame, (1000, 600))
        cv2.imshow('Processed Video', frame_resized)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Processed video saved: {output_video_path}")


# ==================== Run Processing ====================
input_video = os.path.join('videos','mine2.mp4')  # Replace with your input video file
output_video = "final_processed_video.mp4"

process_video(input_video, output_video)
