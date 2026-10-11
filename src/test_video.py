import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import torch
import torch.nn as nn
from create_windows import create_windows
from extract_pose import extract_keypoints

# =========================
# Model Definition (MUST match training)
# =========================

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=99, hidden_size=64, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]   # last time step
        out = self.dropout(out)
        return self.fc(out)   # logits

# =========================
# Load Model
# =========================

device = torch.device("cpu")

model = LSTMModel().to(device)
model.load_state_dict(torch.load("lstm_model.pth", map_location=device))
model.eval()

print("✅ Model Loaded Successfully")

# =========================
# Load Video
# =========================

video_path = "input_videon1.mp4"  # 🔥 change filename if needed

print("📹 Processing Video:", video_path)

keypoints = extract_keypoints(video_path)

if keypoints is None or len(keypoints) < 30:
    print("❌ Video too short or no pose detected.")
    exit()

windows = create_windows(keypoints, window_size=30)

if windows is None or len(windows) == 0:
    print("❌ Not enough motion data.")
    exit()

X_test = torch.tensor(windows, dtype=torch.float32).to(device)

# =========================
# NORMALIZATION (IMPORTANT)
# =========================

# If you saved training mean/std, load them instead
# mean = torch.tensor(np.load("mean.npy"), dtype=torch.float32).to(device)
# std = torch.tensor(np.load("std.npy"), dtype=torch.float32).to(device)

# For now, normalize per video (safer than raw input)
mean = X_test.mean()
std = X_test.std()

X_test = (X_test - mean) / (std + 1e-6)

# =========================
# Prediction
# =========================

with torch.no_grad():
    outputs = model(X_test)              # logits
    probs = torch.sigmoid(outputs)       # convert to probability

probs = probs.cpu().numpy().flatten()

# =========================
# Debug Information
# =========================

print("\nWindow Probabilities:")
print(np.round(probs, 3))

average_score = float(np.mean(probs))
positive_ratio = float((probs > 0.85).sum() / len(probs))

print("\n📊 Autism Score (Average):", round(average_score, 4))
print("📊 Strong Positive Window Ratio:", round(positive_ratio, 3))

# =========================
# STABLE FINAL DECISION
# =========================

if average_score > 0.85 and positive_ratio > 0.7:
    print("\n🟥 Final Prediction: AUTISM DETECTED")
else:
    print("\n🟩 Final Prediction: NORMAL")