import os
import json
import requests
import cv2
import numpy as np
import yt_dlp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Import custom architectures
from train import train
from dataset import TVSumDataset
from utils import  extract_and_fuse_features_from_video
from model import VideoSummarizationModel
from inference import generate_summary

# Target Configurations
FEATURES_DIR = 'data/features'
ANNOTATIONS_DIR = 'data/annotations'
RAW_VIDEOS_DIR = 'data/raw_videos'
EXTRACTED_FRAMES_DIR = 'data/extracted_frames'
TVSUM_DUMMY_ANNOTATIONS_FILENAME = 'tvsum_dummy_annotations.json'

os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
os.makedirs(RAW_VIDEOS_DIR, exist_ok=True)
os.makedirs(EXTRACTED_FRAMES_DIR, exist_ok=True)

# Placeholder objects for model configurations
feature_extractor = None  
motion_target_size = (224, 224)

# 1. Video list configurations
TVSUM_VIDEO_URLS = [
    'https://www.youtube.com/watch?v=dQw4w9WgXcQ', 
    'https://www.youtube.com/watch?v=aqz-KE-bpKQ'  
]

# 2. Download sequence
for i, video_url in enumerate(TVSUM_VIDEO_URLS):
    ydl_opts = {
        'outtmpl': os.path.join(RAW_VIDEOS_DIR, '%(title)s.%(ext)s'),
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True  
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
    except Exception as e:
        print(f"Skipping live download: {e}")

# 3. Handle Ground Truth JSON structure
TVSUM_ANNOTATIONS_URL = "https://raw.githubusercontent.com/yolish/TVSum/master/data/ydata-tvsum50.json"
TVSUM_ANNOTATIONS_FILENAME = 'tvsum_video_info.json'
save_path_annotations = os.path.join(ANNOTATIONS_DIR, TVSUM_ANNOTATIONS_FILENAME)

try:
    response = requests.get(TVSUM_ANNOTATIONS_URL, stream=True)
    response.raise_for_status()
    with open(save_path_annotations, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
except Exception as e:
    print(f"Ground-truth mirror offline, proceeding with pipeline generators: {e}")


# 4. Dummy pipeline generation
def create_dummy_video_from_frames(output_video_path, num_frames=10, width=200, height=200, fps=1.0):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        return
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = colors[i % len(colors)]
        out.write(frame)
    out.release()

dummy_rick_astley_path = os.path.join(RAW_VIDEOS_DIR, 'dummy_rick_astley.mp4')
dummy_big_buck_bunny_path = os.path.join(RAW_VIDEOS_DIR, 'dummy_big_buck_bunny.mp4')

create_dummy_video_from_frames(dummy_rick_astley_path, num_frames=15, fps=2.0) 
create_dummy_video_from_frames(dummy_big_buck_bunny_path, num_frames=20, fps=1.5) 

sample_video_paths = [dummy_rick_astley_path, dummy_big_buck_bunny_path]

tvsum_metadata_for_processing = {}
for video_path in sample_video_paths:
    video_id = os.path.splitext(os.path.basename(video_path))
    tvsum_metadata_for_processing[video_id] = {
        'url': f'placeholder_url_for_{video_id}',
        'description': f'Dummy description for {video_id}'
    }

# Feature Extraction Loop
for video_id, metadata in tvsum_metadata_for_processing.items():
    video_path = os.path.join(RAW_VIDEOS_DIR, f"{video_id}.mp4")
    if not os.path.exists(video_path):
        continue

    output_filename = f'{video_id}_fused_features.pt'
    fused_features = extract_and_fuse_features_from_video(
        video_path=video_path,
        feature_extractor_model=feature_extractor,
        output_filename=output_filename,
        motion_target_size=motion_target_size
    )

# Target annotation processing
tvsum_dummy_annotations = {}
save_path_dummy_annotations = os.path.join(ANNOTATIONS_DIR, TVSUM_DUMMY_ANNOTATIONS_FILENAME)

for video_path in sample_video_paths:
    video_title_no_ext = os.path.splitext(os.path.basename(video_path))
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 10
    cap.release()

    dummy_gt_score = [1.0] * total_frames
    tvsum_dummy_annotations[video_title_no_ext] = {
        'gt_score': dummy_gt_score,
        'url': f'placeholder_url_{video_title_no_ext}'
    }

with open(save_path_dummy_annotations, 'w') as f:
    json.dump(tvsum_dummy_annotations, f, indent=4)

# 5. Pipeline Loader Configuration & Training
tvsum_dataset = TVSumDataset(FEATURES_DIR, ANNOTATIONS_DIR, TVSUM_DUMMY_ANNOTATIONS_FILENAME)

total_samples = len(tvsum_dataset)
if total_samples >= 2:
    train_size = int(0.5 * total_samples)
    val_size = total_samples - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(tvsum_dataset, [train_size, val_size], generator=generator)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = VideoSummarizationModel(input_size=1024, hidden_size=256)
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Invoke your training script logic
    train(5, train_dataloader, optimizer, loss_function, model, val_dataloader)
else:
    print("Insufficient data generated to pass down splits safely.")

# --- 6. Testing Framework for generate_summary ---
print("\n--- Testing generate_summary function ---")

# Test Case 1: 50% ratio on 10 elements selects top 5 highest scores
dummy_scores_1 = np.array([0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.5, 0.6, 0.4, 0.0])
summary_1 = generate_summary(dummy_scores_1, 0.5)
expected_1 = np.array(0.9, 0.8, 0.7, 0.6, 0.5, dtype=np.int8) # Corresponds to: 0.9, 0.8, 0.7, 0.6, 0.5
assert np.array_equal(summary_1, expected_1), "Test 1 Failed"

# Test Case 2: 1/3 ratio on 6 elements selects top 2 highest scores
dummy_scores_2_tensor = torch.tensor([0.2, 0.1, 0.9, 0.8, 0.3, 0.7])
summary_2 = generate_summary(dummy_scores_2_tensor, 1/3)
expected_2 = np.array(0.9, 0.8, dtype=np.int8) # Corresponds to: 0.9, 0.8
assert np.array_equal(summary_2, expected_2), "Test 2 Failed"

print("All assertion evaluation tests passed successfully!")