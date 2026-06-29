import numpy as np
import os
from utils import *
from imageio.v2 import imageio
import json
from torch.utils.data import Dataset
import torch

RESOLUTION = (64, 64)

COLORS = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'purple': (128, 0, 128)
}

OUTPUT_DIR = 'synthetic_videos'

def generate_synthetic_videos(num_videos=10, fps=5):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    video_metadata = []

    shape_types = ['circle', 'square', 'triangle']
    color_names = list(COLORS.keys())
    animation_types = ['translational', 'size_change', 'color_change']

    for i in range(num_videos):
        shape_type = np.random.choice(shape_types)
        start_color_name = np.random.choice(color_names)
        animation_type = np.random.choice(animation_types)

        end_color_name = None
        if animation_type == 'color_change':
            available_end_colors = [c for c in color_names if c != start_color_name]
            if available_end_colors:
                end_color_name = np.random.choice(available_end_colors)
            else:
                # Fallback if only one color exists, prevent infinite loop
                animation_type = np.random.choice([at for at in animation_types if at != 'color_change'])

        num_frames = np.random.randint(10, 21) # 10 to 20 frames

        frames, caption = get_animation_frames(
            shape_type=shape_type,
            start_color_name=start_color_name,
            end_color_name=end_color_name,
            animation_type=animation_type,
            num_frames=num_frames
        )

        video_filename = os.path.join(OUTPUT_DIR, f'video_{i:03d}.gif')
        imageio.mimsave(video_filename, frames, fps=fps) # Save as GIF

        video_metadata.append({'video_path': video_filename, 'caption': caption})

    print(f"Generated {num_videos} synthetic videos in '{OUTPUT_DIR}'")
    return video_metadata

class VideoDataset(Dataset):
  def __init__(self,metadata_path,vocab):
    with open(metadata_path,'r')as f:
      self.metadata=json.load(f)
    self.vocab=vocab
  def __len__(self):
    return len(self.metadata)
  def __getitem__(self,index):
    item=self.metadata[index]
    video_path=item['video_path']
    caption_str=item['caption']
    frames_np=np.array(imageio.mimread(video_path))
    video_tensor=torch.from_numpy(frames_np).float()/255.0
    video_tensor=video_tensor.permute(3,0,1,2)
    caption_sequence=text_to_sequence(caption_str,self.vocab)
    return video_tensor,caption_sequence
  print("VideoDataset class defined.")