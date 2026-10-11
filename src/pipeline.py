import os
import numpy as np
from extract_pose import extract_keypoints
from create_windows import create_windows

CLIP_FOLDER = "data/clips"

X_all = []
y_all = []

failed_clips = []

for file in os.listdir(CLIP_FOLDER):

    if not file.endswith(".mp4"):
        continue

    path = os.path.join(CLIP_FOLDER, file)
    print(f"Processing {file}...")

    try:
        keypoints = extract_keypoints(path)

        if keypoints is None or len(keypoints) < 30:
            print("⚠️ Skipped (too short or empty)")
            failed_clips.append(file)
            continue

        windows = create_windows(keypoints, window_size=30)

        if windows is None or len(windows) == 0:
            print("⚠️ Skipped (no windows created)")
            failed_clips.append(file)
            continue

        # 🔍 Safety shape check
        if len(windows.shape) != 3:
            print("⚠️ Skipped (wrong shape)", windows.shape)
            failed_clips.append(file)
            continue

        X_all.append(windows)

        if "positive" in file:
            y_all.append(np.ones(len(windows)))
        else:
            y_all.append(np.zeros(len(windows)))

    except Exception as e:
        print("❌ Error processing:", file)
        print("Reason:", e)
        failed_clips.append(file)
        continue


print("\nBuilding final dataset...")

if len(X_all) == 0:
    print("❌ No valid data found.")
    exit()

X_final = np.concatenate(X_all, axis=0)
y_final = np.concatenate(y_all, axis=0)

np.save("X_dataset.npy", X_final)
np.save("y_dataset.npy", y_final)

print("\nDataset saved successfully!")
print("Final X shape:", X_final.shape)
print("Final y shape:", y_final.shape)

if failed_clips:
    print("\nSkipped clips:")
    for f in failed_clips:
        print(f)