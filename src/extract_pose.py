import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_PATH = "pose_landmarker_lite.task"

def extract_keypoints(video_path):
    # Download model if not present
    if not os.path.exists(MODEL_PATH):
        print("Downloading pose landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")

    # Create pose landmarker
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO)
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(video_path)
        keypoints = []
        frame_timestamp_ms = 0
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            # Process every 3rd frame for speed
            if frame_count % 3 != 0:
                frame_timestamp_ms += 33  # still increment timestamp
                continue
                
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            frame_timestamp_ms += 33 * 3  # increment by 3 frames worth
            
            if result.pose_landmarks:
                frame_points = []
                for lm in result.pose_landmarks[0]:
                    frame_points.extend([lm.x, lm.y, lm.z])
                keypoints.append(frame_points)
                
        cap.release()
        if len(keypoints) == 0:
            return None
        return np.array(keypoints)