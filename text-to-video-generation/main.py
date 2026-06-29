from dataset import generate_synthetic_videos,OUTPUT_DIR,RESOLUTION,VideoDataset
import os
from model import TextToVideoModel
import json
from utils import build_vocabulary,text_to_sequence,collate_fn
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import numpy as np
from imageio.v2 import imageio

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
synthetic_dataset = generate_synthetic_videos(num_videos=5)

METADATA_FILENAME = 'metadata.json'

# Save the metadata to a JSON file
with open(os.path.join(OUTPUT_DIR, METADATA_FILENAME), 'w') as f:
    json.dump(synthetic_dataset, f, indent=4)

metadata_path=os.path.join(OUTPUT_DIR,METADATA_FILENAME)

with open(metadata_path,'r')as f:
  loaded_metadata=json.load(f)

vocabulary = build_vocabulary(loaded_metadata)

VOCAB_SIZE = len(vocabulary)
EMBEDDING_DIM = 64
HIDDEN_SIZE = 128 # Output size of TextEncoder
VIDEO_CHANNELS = 3 # RGB
VIDEO_FRAMES = 16 # Target number of frames
VIDEO_HEIGHT, VIDEO_WIDTH = RESOLUTION # 64x64

model = TextToVideoModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE,
                         VIDEO_CHANNELS, VIDEO_FRAMES, VIDEO_HEIGHT, VIDEO_WIDTH)

import torch.optim as optim
criterion=nn.MSELoss()
learning_rate=0.001
optimizer=optim.Adam(model.parameters(),lr=learning_rate)
dataset = VideoDataset(metadata_path, vocabulary)

BATCH_SIZE=2
data_loader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,collate_fn=collate_fn)
model.eval()

# 1. Define a new dummy text input
dummy_text_input = 'A red square moves from left to right.'

# 2. Convert to numerical sequence
dummy_caption_sequence = text_to_sequence(dummy_text_input, vocabulary)

# 3. Add batch dimension and move to device
dummy_caption_sequence = dummy_caption_sequence.unsqueeze(0).to(device)

print(f"Generating video for caption: '{dummy_text_input}'")

# 4. Generate video frames with torch.no_grad()
with torch.no_grad():
    generated_video = model(dummy_caption_sequence)

# 5. Post-process the generated video tensor
# Move to CPU and convert to numpy
generated_video_np = generated_video.squeeze(0).cpu().numpy()

# De-normalize from [-1, 1] to [0, 255] and convert to uint8
generated_video_np = ((generated_video_np + 1) / 2 * 255).astype(np.uint8)

# Adjust channel order from [C, T, H, W] to [T, H, W, C]
generated_video_np = generated_video_np.transpose(1, 2, 3, 0) # (T, H, W, C)

# 6. Save the generated frames as a GIF file
generated_video_path = os.path.join(OUTPUT_DIR, 'generated_video.gif')
imageio.mimsave(generated_video_path, generated_video_np, fps=5)

print(f"Generated video saved to '{generated_video_path}'")
