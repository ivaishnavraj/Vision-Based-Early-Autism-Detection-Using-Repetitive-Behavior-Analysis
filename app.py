from flask import Flask, request, render_template, redirect, url_for, flash
import os
import tempfile
import torch
import torch.nn as nn
import numpy as np
from werkzeug.utils import secure_filename

import sys
sys.path.append('src')

# Import existing modules
from extract_pose import extract_keypoints
from create_windows import create_windows

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
app.config['VIDEO_FOLDER'] = os.path.join(app.root_path, 'static', 'videos')
os.makedirs(app.config['VIDEO_FOLDER'], exist_ok=True)

# Model Definition (same as in test_video.py)
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=99, hidden_size=64, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)

# Load model
device = torch.device("cpu")
model = LSTMModel().to(device)
model.load_state_dict(torch.load("lstm_model.pth", map_location=device))
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Clean up old videos (keep only last 5)
    video_files = sorted([f for f in os.listdir(app.config['VIDEO_FOLDER']) if f.endswith('.mp4')], 
                        key=lambda x: os.path.getctime(os.path.join(app.config['VIDEO_FOLDER'], x)))
    if len(video_files) > 5:
        for old_file in video_files[:-5]:
            os.remove(os.path.join(app.config['VIDEO_FOLDER'], old_file))

    if 'video' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['video']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and file.filename.lower().endswith('.mp4'):
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['VIDEO_FOLDER'], filename)
        file.save(video_path)

        # Process the video
        results = process_video(video_path)

        # Add video path to results
        results['video_path'] = f'/static/videos/{filename}'

        return render_template('results.html', results=results)
    else:
        flash('Invalid file type. Please upload an MP4 video.')
        return redirect(url_for('index'))

def process_video(video_path):
    keypoints = extract_keypoints(video_path)
    if keypoints is None or len(keypoints) < 30:
        return {"error": "Video too short or no pose detected."}

    # Use larger step size for performance (every 15 frames instead of 1)
    windows = create_windows(keypoints, window_size=30, step_size=15)
    if windows is None or len(windows) == 0:
        return {"error": "No windows created from video."}

    # Convert to tensor
    X_test = torch.tensor(windows, dtype=torch.float32).to(device)

    # Normalize per video (same as test_video.py)
    mean = X_test.mean()
    std = X_test.std()
    X_test = (X_test - mean) / (std + 1e-6)

    # Predict
    with torch.no_grad():
        outputs = model(X_test)
        predictions = torch.sigmoid(outputs).cpu().numpy().flatten()

    # Use lower threshold (maybe the model outputs need adjustment)
    positive_count = np.sum(predictions > 0.3)  # Lower threshold
    total_windows = len(predictions)

    return {
        "total_windows": total_windows,
        "positive_count": int(positive_count),
        "percentage": float(positive_count / total_windows * 100) if total_windows > 0 else 0,
        "predictions": predictions.tolist()
    }

if __name__ == '__main__':
    app.run(debug=True)