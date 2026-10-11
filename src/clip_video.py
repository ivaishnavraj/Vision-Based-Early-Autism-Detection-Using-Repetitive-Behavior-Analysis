import subprocess
import os


def clip_video(input_path, start_time, end_time, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-ss", start_time,
        "-to", end_time,
        "-i", input_path,
        "-c", "copy",
        output_path
    ]

    subprocess.run(command)