import os
import numpy as np
from extract_pose import extract_keypoints
from create_windows import create_windows

CLIP_FOLDER = "data/clips"

X_all = []
y_all = []

for file in os.listdir(CLIP_FOLDER):

    if not file.endswith(".mp4"):
        continue

    path = os.path.join(CLIP_FOLDER, file)
    print(f"Processing {file}...")

    keypoints = extract_keypoints(path)

    if keypoints is None or len(keypoints) < 30:
        continue

    windows = create_windows(keypoints, window_size=30)

    if windows is None or len(windows) == 0:
        continue

    X_all.append(windows)

    if "positive" in file:
        y_all.append(np.ones(len(windows)))
    else:
        y_all.append(np.zeros(len(windows)))

print("Building final dataset...")

X_final = np.concatenate(X_all, axis=0)
y_final = np.concatenate(y_all, axis=0)

np.save("X_dataset.npy", X_final)
np.save("y_dataset.npy", y_final)

print("Dataset saved successfully.")
print("Final shape:", X_final.shape)