import os
import yt_dlp


def download_video(url, output_folder="data/raw_videos", filename=None):
    os.makedirs(output_folder, exist_ok=True)

    if filename:
        output_path = os.path.join(output_folder, filename)
    else:
        output_path = os.path.join(output_folder, "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "mp4",
        "outtmpl": output_path,
        "quiet": False
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])