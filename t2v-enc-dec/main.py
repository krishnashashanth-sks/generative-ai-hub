import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import cv2
from model import VideoTextGenerator
from dataset import VideoTextDataset
from train import train_model
import re
import torch.nn as nn
import torch.optim as optim

# Define parameters for dummy video generation
num_dummy_videos = 2  # Number of dummy video files to create
video_duration_frames = 50  # Length of each video in frames
frame_width = 128  # Width of each video frame
frame_height = 128  # Height of each video frame
frame_rate = 10  # Frames per second (FPS)

print(f"Dummy video parameters defined:\n  Number of videos: {num_dummy_videos}\n  Duration per video: {video_duration_frames} frames\n  Resolution: {frame_width}x{frame_height}\n  Frame Rate: {frame_rate} FPS")

dummy_videos_dir = "dummy_videos"
os.makedirs(dummy_videos_dir, exist_ok=True)
print(f"Directory '{dummy_videos_dir}' ensured to exist for storing dummy videos.")

print("Generating dummy video files...")

dummy_video_paths = []

for i in range(num_dummy_videos):
    video_filename = os.path.join(dummy_videos_dir, f'dummy_video_{i+1}.mp4')
    dummy_video_paths.append(video_filename)

    # Define the codec and create VideoWriter object
    # Use XVID for wider compatibility, or MP4V if XVID doesn't work
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # You can also try 'XVID'
    out = cv2.VideoWriter(video_filename, fourcc, frame_rate, (frame_width, frame_height))

    if not out.isOpened():
        print(f"Error: Could not open video writer for {video_filename}")
        continue

    for frame_idx in range(video_duration_frames):
        # Create a blank frame
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        # Simple visual pattern: a moving colorful rectangle
        # The color and position change with frame_idx
        color = (255 * (frame_idx / video_duration_frames),
                 255 * ((frame_idx + video_duration_frames/3) / video_duration_frames % 1),
                 255 * ((frame_idx + video_duration_frames*2/3) / video_duration_frames % 1))

        # Ensure color values are within 0-255
        color = tuple(int(c) for c in color)

        # Rectangle moves from left to right
        rect_width = frame_width // 4
        rect_height = frame_height // 4
        x_pos = int((frame_idx / video_duration_frames) * (frame_width - rect_width))
        y_pos = frame_height // 2 - rect_height // 2

        cv2.rectangle(frame, (x_pos, y_pos), (x_pos + rect_width, y_pos + rect_height), color, -1)

        # Add a pulsating circle in the center
        radius = int(20 + 15 * np.sin(2 * np.pi * frame_idx / video_duration_frames))
        cv2.circle(frame, (frame_width // 2, frame_height // 2), radius, (0, 255, 255), -1)

        out.write(frame)

    out.release()
    print(f"Generated dummy video: {video_filename}")

print("Dummy video generation complete.")
print(f"Generated video paths: {dummy_video_paths}")

frames_dir = "dummy_frames"
os.makedirs(frames_dir, exist_ok=True)
print(f"Directory '{frames_dir}' ensured to exist for storing extracted frames.")

print("Starting frame extraction from dummy videos...")

for video_path in dummy_video_paths:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video_frames_dir = os.path.join(frames_dir, video_name)
    os.makedirs(output_video_frames_dir, exist_ok=True)
    print(f"Created directory for frames: {output_video_frames_dir}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        continue

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(output_video_frames_dir, f'frame_{frame_count:04d}.png')
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames from {video_path}")

print("Frame extraction complete.")

target_width = 64
target_height = 64
normalization_range = "[-1, 1]" # Can be "[0, 1]" or "[-1, 1]"

preprocessed_frames_dir = "preprocessed_frames"
os.makedirs(preprocessed_frames_dir, exist_ok=True)
print(f"Directory '{preprocessed_frames_dir}' ensured to exist for storing preprocessed frames.")
print("Starting frame preprocessing...")


# Helper function for normalization
def normalize_frame(frame, norm_range):
    frame_float = frame.astype(np.float32) # Convert to float32
    if norm_range == "[0, 1]":
        return frame_float / 255.0
    elif norm_range == "[-1, 1]":
        return (frame_float / 127.5) - 1.0
    else:
        raise ValueError("Unsupported normalization range. Use '[0, 1]' or '[-1, 1]'")

# Iterate through each video's frames in the dummy_frames directory
for video_name_dir in os.listdir(frames_dir):
    video_frames_path = os.path.join(frames_dir, video_name_dir)
    if not os.path.isdir(video_frames_path):
        continue

    output_preprocessed_video_dir = os.path.join(preprocessed_frames_dir, video_name_dir)
    os.makedirs(output_preprocessed_video_dir, exist_ok=True)
    print(f"Processing frames for video: {video_name_dir}")

    frame_files = sorted([f for f in os.listdir(video_frames_path) if f.endswith('.png')])

    for frame_file in frame_files:
        # Load the frame
        frame_path = os.path.join(video_frames_path, frame_file)
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"Warning: Could not load frame {frame_path}. Skipping.")
            continue

        # Resize the frame
        resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

        # Apply pixel value normalization
        normalized_frame = normalize_frame(resized_frame, normalization_range)

        # Save the preprocessed frame (as a .npy file to preserve float values easily)
        # For visual inspection, one might save as PNG, but for model input, .npy is better
        output_frame_path_npy = os.path.join(output_preprocessed_video_dir, os.path.splitext(frame_file)[0] + '.npy')
        np.save(output_frame_path_npy, normalized_frame)

        # Optionally, save a PNG for visual verification (uncomment if needed)
        # output_frame_path_png = os.path.join(output_preprocessed_video_dir, frame_file)
        # cv2.imwrite(output_frame_path_png, ((normalized_frame + 1) * 127.5).astype(np.uint8)) # Convert back to 0-255 for PNG

print("Frame preprocessing complete.")

full_vocabulary = ['<PAD>', '<UNK>', '<SOS>', '<EOS>', 'a', 'and', 'are', 'center', 'circle', 'clip', 'colorful', 'dynamic', 'from', 'in', 'left', 'moves', 'pulsating', 'rectangle', 'right', 'short', 'shown', 'small', 'the', 'this', 'to', 'with', 'yellow']
word_to_id = {word: i for i, word in enumerate(full_vocabulary)}
pad_token_id = word_to_id['<PAD>']
unk_token_id = word_to_id['<UNK>']
sos_token_id = word_to_id['<SOS>']
eos_token_id = word_to_id['<EOS>']

# Define the custom collate_fn (from cell 4c65ca8f)
def collate_fn(batch):
    video_sequences = [item['video'] for item in batch]
    text_sequences = [item['text'] for item in batch]

    batched_videos = torch.stack([torch.from_numpy(video).float() for video in video_sequences])

    max_batch_seq_len = max(len(seq) for seq in text_sequences)
    padded_text_sequences = []
    for seq in text_sequences:
        # pad_token_id (0) is assumed to be defined from previous steps
        padded_seq = seq + [pad_token_id] * (max_batch_seq_len - len(seq))
        padded_text_sequences.append(padded_seq)

    batched_text = torch.LongTensor(padded_text_sequences)

    return {"video": batched_videos, "text": batched_text}

dummy_video_descriptions = [
    "A colorful rectangle moves from left to right with a pulsating yellow circle in the center.",
    "A small colorful rectangle and a dynamic yellow circle are shown in this short clip."
]
def custom_tokenizer(text, word_to_id_map, unk_id, sos_id, eos_id):
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    token_ids = [sos_id] # Start with SOS token
    for word in words:
        token_ids.append(word_to_id_map.get(word, unk_id)) # Map word to ID, default to UNK
    token_ids.append(eos_id) # End with EOS token
    return token_ids

tokenized_descriptions = []
for desc in dummy_video_descriptions:
    token_sequence = custom_tokenizer(desc, word_to_id, unk_token_id, sos_token_id, eos_token_id)
    tokenized_descriptions.append(token_sequence)


dataset = VideoTextDataset(preprocessed_frames_dir, tokenized_descriptions)
batch_size = 2 # from training hyperparameters
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Parameters for model instantiation (from previous optimization step, cell 723a22a6)
vocab_size = 27
embedding_dim = 128
hidden_size = 256
in_channels = 3
latent_dim = 512
num_frames = 50
frame_height = 64
frame_width = 64

# Instantiate the videotext_generator model with the optimized architecture
videotext_generator = VideoTextGenerator(
    vocab_size=vocab_size,
    text_embedding_dim=embedding_dim,
    text_hidden_size=hidden_size,
    video_in_channels=in_channels,
    video_latent_dim=latent_dim,
    num_frames=num_frames,
    frame_height=frame_height,
    frame_width=frame_width
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
videotext_generator.to(device)

criterion = nn.MSELoss()

learning_rate = 0.001

optimizer = optim.Adam(videotext_generator.parameters(), lr=learning_rate)

num_epochs=5

train_model(num_epochs,train_dataloader,videotext_generator,optimizer,criterion,device)

# Set the model to evaluation mode
videotext_generator.eval()

# Prepare to store a few examples for visualization
example_ground_truth_videos = []
example_generated_videos = []
example_text_descriptions = []

# Iterate through the dataloader to get some samples
print("Fetching examples for qualitative evaluation with optimized model...")
with torch.no_grad():
    for batch_idx, batch in enumerate(train_dataloader):
        if len(example_ground_truth_videos) >= 2: # Get 2 examples for visualization
            break

        video_sequence = batch['video'].to(device)
        text_sequence = batch['text'].to(device)

        # Forward pass to generate video
        generated_video = videotext_generator(video_sequence, text_sequence)

        # Move to CPU and convert to numpy for visualization
        gt_video_np = video_sequence.cpu().numpy()
        gen_video_np = generated_video.cpu().numpy()

        # Store the first sample from the batch
        example_ground_truth_videos.append(gt_video_np[0])
        example_generated_videos.append(gen_video_np[0])

        # Decode the text sequence
        decoded_text = []
        for token_id in text_sequence[0].cpu().numpy():
            if token_id not in [pad_token_id, sos_token_id, eos_token_id]:
                decoded_text.append(full_vocabulary[token_id])
        example_text_descriptions.append(" ".join(decoded_text))

print("Starting qualitative visual inspection of optimized model...")

# Helper function to denormalize and convert to uint8 for display
def denormalize_and_convert_to_uint8(video_frames):
    # video_frames is assumed to be in [-1, 1] range and float32
    # Convert to [0, 255] and then to uint8
    denormalized_frames = ((video_frames + 1) * 127.5).astype(np.uint8)
    return denormalized_frames

# Visualize a few frames from each collected example
num_frames_to_display = 5 # Display 5 frames per video

for i in range(len(example_ground_truth_videos)):
    gt_video = denormalize_and_convert_to_uint8(example_ground_truth_videos[i])
    gen_video = denormalize_and_convert_to_uint8(example_generated_videos[i])
    text_desc = example_text_descriptions[i]

    print(f"\n--- Visualizing Example {i+1} (Optimized Model) ---")
    print(f"Text Description: {text_desc}")

    fig, axes = plt.subplots(num_frames_to_display, 2, figsize=(8, num_frames_to_display * 4))
    fig.suptitle(f'Optimized Model Example {i+1} - Description: {text_desc[:50]}...', fontsize=16)

    # Select frames to display evenly spaced across the video duration
    frame_indices = np.linspace(0, gt_video.shape[0] - 1, num_frames_to_display, dtype=int)

    for j, frame_idx in enumerate(frame_indices):
        # Ground Truth Frame
        ax_gt = axes[j, 0]
        ax_gt.imshow(gt_video[frame_idx])
        ax_gt.set_title(f'GT Frame {frame_idx}')
        ax_gt.axis('off')

        # Generated Frame
        ax_gen = axes[j, 1]
        ax_gen.imshow(gen_video[frame_idx])
        ax_gen.set_title(f'Generated Frame {frame_idx}')
        ax_gen.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

print("Qualitative visual inspection of optimized model complete.")

print("Starting quantitative evaluation (MSE calculation) for optimized model...")

# Re-initialize total_eval_loss to ensure it's fresh for this calculation
total_eval_eval_loss = 0.0
num_eval_batches = 0

# Ensure the model is in evaluation mode (already set in previous step, but good practice)
videotext_generator.eval()

# Use a separate criterion for evaluation if needed, or reuse the training one
eval_criterion = nn.MSELoss()

with torch.no_grad():
    # Iterate through the dataloader to calculate metrics over the whole dataset
    # For this dummy setup, train_dataloader has all data, but normally you'd use a separate eval_dataloader
    for batch_idx, batch in enumerate(train_dataloader):
        video_sequence = batch['video'].to(device)
        text_sequence = batch['text'].to(device)

        generated_video = videotext_generator(video_sequence, text_sequence)

        loss = eval_criterion(generated_video, video_sequence)
        total_eval_eval_loss += loss.item()
        num_eval_batches += 1

    average_mse_optimized_model = total_eval_eval_loss / num_eval_batches
    print(f"Quantitative Evaluation for Optimized Model Complete.\nAverage MSE Loss: {average_mse_optimized_model:.4f}")
