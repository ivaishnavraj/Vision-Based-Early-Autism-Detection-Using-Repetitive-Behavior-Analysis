import numpy as np

def create_windows(data, window_size=30, step_size=1):
    """
    Create sliding windows from pose sequence.

    data: numpy array of shape (frames, features)
    window_size: number of frames per window
    step_size: sliding step
    """

    windows = []

    if len(data) < window_size:
        return None

    for i in range(0, len(data) - window_size + 1, step_size):
        window = data[i:i+window_size]
        windows.append(window)

    if len(windows) == 0:
        return None

    return np.array(windows)