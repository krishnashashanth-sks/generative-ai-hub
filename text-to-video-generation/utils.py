from PIL import Image, ImageDraw
from dataset import RESOLUTION,COLORS
import numpy as np
import random
import torch.nn.functional as F
from vocabulary import Vocabulary
import torch
from main import vocabulary
from collections import Counter

def collate_fn(batch):
  video_tensors=[item[0] for item in batch]
  caption_sequences=[item[1] for item in batch]
  max_frames=max([v.shape[1] for v in video_tensors])
  padded_video_tensors=[]
  for video_tensor in video_tensors:
    num_frames=video_tensor.shape[1]
    padding_needed=max_frames-num_frames
    if padding_needed>0:
      padded_video_tensor=F.pad(video_tensor,(0,0,0,0,0,padding_needed,0,0),'constant',0)
    else:
      padded_video_tensor=video_tensor
    padded_video_tensors.append(padded_video_tensor)
  padded_caption_sequences=torch.nn.utils.rnn.pad_sequence(
      caption_sequences,
      batch_first=True,
      padding_value=vocabulary.word2idx['<pad>']
  )
  video_batch=torch.stack(padded_video_tensors)
  return video_batch,padded_caption_sequences

print("collate_fn defined")
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def draw_shape(draw, shape_type, color, position, size):
    # Position is the center of the shape
    x, y = position
    w, h = size

    if shape_type == 'circle':
        draw.ellipse([(x - w // 2, y - h // 2), (x + w // 2, y + h // 2)], fill=color)
    elif shape_type == 'square':
        draw.rectangle([(x - w // 2, y - h // 2), (x + w // 2, y + h // 2)], fill=color)
    elif shape_type == 'triangle':
        # For simplicity, an equilateral triangle pointing up
        points = [
            (x, y - h // 2), # Top point
            (x - w // 2, y + h // 2), # Bottom-left point
            (x + w // 2, y + h // 2)  # Bottom-right point
        ]
        draw.polygon(points, fill=color)

def interpolate_color(start_color, end_color, t):
    """Interpolates between two RGB colors based on a factor t (0 to 1)."""
    r = int(start_color[0] + t * (end_color[0] - start_color[0]))
    g = int(start_color[1] + t * (end_color[1] - start_color[1]))
    b = int(start_color[2] + t * (end_color[2] - start_color[2]))
    return (r, g, b)

def get_animation_frames(shape_type, start_color_name, end_color_name=None, animation_type='static', num_frames=15):
    frames = []
    width, height = RESOLUTION

    # Initial properties
    color = COLORS[start_color_name]
    initial_size = 20 # A reasonable starting size for a 64x64 frame

    # Determine animation specifics
    if animation_type == 'translational':
        # Randomize movement direction (left-right, top-bottom, diagonal)
        movement_patterns = [
            ((initial_size // 2, height // 2), (width - initial_size // 2, height // 2), "from left to right"),
            ((width // 2, initial_size // 2), (width // 2, height - initial_size // 2), "from top to bottom"),
            ((initial_size // 2, initial_size // 2), (width - initial_size // 2, height - initial_size // 2), "diagonally from top-left to bottom-right"),
            ((width - initial_size // 2, initial_size // 2), (initial_size // 2, height - initial_size // 2), "diagonally from top-right to bottom-left")
        ]

        start_pos, end_pos, movement_description = random.choice(movement_patterns) # Fixed: using random.choice

        caption = f"A {start_color_name} {shape_type} moves {movement_description}."

    elif animation_type == 'size_change':
        start_size = 15
        end_size = 35 if np.random.rand() > 0.5 else 5 # Either grows or shrinks
        description = 'grows larger' if end_size > start_size else 'shrinks smaller'
        start_pos = (width // 2, height // 2)
        end_pos = start_pos
        caption = f"A {start_color_name} {shape_type} {description}."

    elif animation_type == 'color_change':
        if not end_color_name or end_color_name == start_color_name:
            # Pick a different random color if end_color is not specified or is same as start
            available_colors = [c for c in COLORS.keys() if c != start_color_name]
            end_color_name = np.random.choice(available_colors)

        start_pos = (width // 2, height // 2)
        end_pos = start_pos
        size = initial_size
        caption = f"A {start_color_name} {shape_type} changes color to {end_color_name}."

    else: # static or no specific animation
        start_pos = (width // 2, height // 2)
        end_pos = start_pos
        caption = f"A {start_color_name} {shape_type} is static."

    for i in range(num_frames):
        img = Image.new('RGB', RESOLUTION, color='black')
        draw = ImageDraw.Draw(img)

        t = i / (num_frames - 1) if num_frames > 1 else 0

        current_pos_x = int(start_pos[0] + t * (end_pos[0] - start_pos[0]))
        current_pos_y = int(start_pos[1] + t * (end_pos[1] - start_pos[1]))
        current_pos = (current_pos_x, current_pos_y)

        if animation_type == 'size_change':
            current_size = int(start_size + t * (end_size - start_size))
            current_shape_size = (current_size, current_size)
            current_color = COLORS[start_color_name]
        elif animation_type == 'color_change':
            current_color = interpolate_color(COLORS[start_color_name], COLORS[end_color_name], t)
            current_shape_size = (initial_size, initial_size)
        else:
            current_color = COLORS[start_color_name]
            current_shape_size = (initial_size, initial_size)

        draw_shape(draw, shape_type, current_color, current_pos, current_shape_size)
        frames.append(np.array(img))

    return frames, caption

def build_vocabulary(metadata,threshold=1):
  counter=Counter()
  for item in metadata:
    caption=item['caption'].lower().replace('.',',').split()
    counter.update(caption)
  words=[word for word,cnt in counter.items() if cnt>=threshold]
  vocab=Vocabulary()
  for word in words:
    vocab.add_word(word)
  return vocab

def text_to_sequence(caption,vocab):
  tokens=caption.lower().replace(',','').split()
  sequence=[vocab('<start')]+[vocab(token) for token in tokens]+[vocab('end')]
  return torch.tensor(sequence,dtype=torch.long)