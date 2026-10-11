from src.extract_pose import PoseExtractor

video_path = "data/clips/positive_clip.mp4"

extractor = PoseExtractor()
data = extractor.extract_from_video(video_path)
extractor.close()

print("Extracted shape:", data.shape)