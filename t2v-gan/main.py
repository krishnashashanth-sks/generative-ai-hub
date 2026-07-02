import numpy as np
from torchvision import transforms
from moviepy.editor import ImageSequenceClip
from IPython.display import HTML
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import cv2
from torchvision import transforms
from transformers import AutoTokenizer
from dataset import VideoTextDataset
from train import train_model
from generator import VideoGenerator
from discriminator import VideoDiscriminator
from text_encoder import TextEncoder
from video_encoder import VideoEncoder

# Placeholder for text descriptions
# In a real scenario, this would be loaded from a JSON or CSV file associated with the dataset
text_descriptions = [
    "A dog running in a park.",
    "A cat playing with a ball of yarn.",
    "A person riding a bicycle down a street.",
    "A bird flying in the sky.",
    "A car driving on a highway."
]

# Placeholder for hypothetical video file paths
# In a real scenario, these would point to actual video files on disk
# We'll simulate creating some dummy video files for demonstration later if needed.
video_paths = [
    "data/videos/video_001.mp4",
    "data/videos/video_002.mp4",
    "data/videos/video_003.mp4",
    "data/videos/video_004.mp4",
    "data/videos/video_005.mp4"
]

# Create a dummy directory for videos to avoid FileNotFoundError later if paths are accessed
os.makedirs('data/videos', exist_ok=True)

# Display the loaded data to confirm
print("--- Hypothetical Text Descriptions ---")
for i, desc in enumerate(text_descriptions):
    print(f"[{i+1}] {desc}")

print("\n--- Hypothetical Video Paths ---")
for i, path in enumerate(video_paths):
    print(f"[{i+1}] {path}")

print("\nPlaceholder data for text descriptions and video paths created.")
# Choose a pre-trained tokenizer (e.g., 'bert-base-uncased' or 'distilbert-base-uncased')
# For text-to-video, a robust general-purpose text encoder is usually a good starting point.
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize the text descriptions
# We'll set a max_length to ensure consistent input size and padding/truncation
max_text_length = 64 # A common length for text descriptions in these datasets

# Tokenize and encode the text descriptions
# 'return_tensors="pt"' returns PyTorch tensors
# 'padding=True' pads to the longest sequence in the batch or to max_length if specified
# 'truncation=True' truncates to max_length if the sequence is longer
encoded_texts = tokenizer(
    text_descriptions,
    padding="max_length",
    truncation=True,
    max_length=max_text_length,
    return_tensors="pt"
)

# The result is a dictionary containing 'input_ids' and 'attention_mask'
input_ids = encoded_texts['input_ids']
attention_mask = encoded_texts['attention_mask']

# --- Parameters for video processing ---
num_frames_per_video = 16  # Number of frames to sample from each video
frame_height, frame_width = 128, 128  # Target resolution for video frames

# --- Step 1: Create dummy video files for demonstration ---
# This is necessary because the video_paths are currently just strings.
# In a real scenario, you would already have actual video files.

def create_dummy_video(filename, duration_seconds=2, fps=10, width=frame_width, height=frame_height):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V') # Codec for .mp4 files
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not open video writer for {filename}")
        return

    for i in range(duration_seconds * fps):
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        # Optionally, make frames change in a discernible way
        color = (i * 10 % 255, (i * 5 + 50) % 255, (i * 2 + 100) % 255)
        cv2.circle(frame, (width // 2, height // 2), min(width, height) // 4, color, -1)
        out.write(frame)
    out.release()

print("Creating dummy video files...")
for path in video_paths:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    create_dummy_video(path)
print("Dummy video files created.")

# --- Step 2: Define video transformations ---
video_transform = transforms.Compose([
    transforms.ToPILImage(), # Convert numpy array to PIL Image for torchvision transforms
    transforms.Resize((frame_height, frame_width)),
    transforms.ToTensor(), # Convert PIL Image to Tensor, scales to [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats for normalization
])

# --- Step 3: Process video frames ---
def process_video(video_path, transform, num_frames):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print(f"Warning: Video {video_path} has 0 frames.")
        return torch.empty(0)

    # Calculate frame indices to sample evenly
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # OpenCV reads BGR, convert to RGB
            frames.append(transform(frame_rgb))

        if len(frames) == num_frames:
            break # Stop if we have enough frames

    cap.release()

    # Pad if not enough frames were sampled (e.g., very short video)
    if len(frames) < num_frames:
        print(f"Warning: Not enough frames sampled for {video_path}. Padding with zeros.")
        # Create a tensor of zeros with the expected shape
        padding_frame = torch.zeros_like(frames[0]) if frames else torch.zeros(3, frame_height, frame_width)
        while len(frames) < num_frames:
            frames.append(padding_frame)

    return torch.stack(frames)

print("Processing video frames...")
processed_video_frames = []
for path in video_paths:
    processed_frames = process_video(path, video_transform, num_frames_per_video)
    processed_video_frames.append(processed_frames)

# Convert list of tensors to a single tensor
# Resulting shape: (num_videos, num_frames_per_video, C, H, W)
processed_video_frames_tensor = torch.stack(processed_video_frames)

video_text_dataset = VideoTextDataset(input_ids, attention_mask, processed_video_frames_tensor)

# Define batch size
batch_size = 2 # Small batch size for demonstration with limited samples

# --- Split dataset into training and (optional) validation sets ---
# For a real scenario, you'd have a much larger dataset to split.
# Here, with only 5 samples, splitting will be very simple or we might just use the full dataset for one DataLoader.
# Let's create a dummy split for illustration purposes.

total_samples = len(video_text_dataset)
if total_samples > 1:
    train_size = max(1, int(0.8 * total_samples)) # At least 1 sample for training
    val_size = total_samples - train_size
    train_dataset, val_dataset = random_split(video_text_dataset, [train_size, val_size])
else:
    train_dataset = video_text_dataset
    val_dataset = None # No validation set if only one sample

print(f"Training dataset size: {len(train_dataset)} samples")
if val_dataset:
    print(f"Validation dataset size: {len(val_dataset)} samples")

# --- Instantiate DataLoader for the training set ---
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,  # Shuffle data for training
    num_workers=0  # For simplicity, 0 workers in Colab/small setup. Increase for larger datasets.
)

# --- Instantiate DataLoader for the validation set (if available) ---
val_dataloader = None
if val_dataset:
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, # No need to shuffle validation data
        num_workers=0
    )
video_encoder = VideoEncoder()


embedding_dim = 768
latent_dim = 100
output_channels = 3
num_frames_per_video = 16
frame_height = 128
frame_width = 128

text_encoder = TextEncoder(vocab_size=tokenizer.vocab_size, embedding_dim=embedding_dim)

# 5. Instantiate the VideoGenerator
video_generator = VideoGenerator(
    text_embedding_dim=embedding_dim,
    latent_dim=latent_dim,
    output_channels=output_channels,
    num_frames=num_frames_per_video,
    frame_height=frame_height,
    frame_width=frame_width
)

video_discriminator = VideoDiscriminator(
    in_channels=output_channels, # From previous step, output_channels is 3 (RGB)
    num_frames=num_frames_per_video,
    frame_height=frame_height,
    frame_width=frame_width
)

# 4. Set up optimizers for Generator and Discriminator
lr = 0.0002 # Learning rate
beta1 = 0.5 # Beta1 for Adam optimizer, common for GANs
beta2 = 0.999 # Beta2 for Adam optimizer

# Assuming video_generator and video_discriminator instances are already defined from previous steps
optimizer_G = optim.Adam(video_generator.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_D = optim.Adam(video_discriminator.parameters(), lr=lr, betas=(beta1, beta2))

device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
print(f"Using device:{device}")

# Ensure models are moved to the device and set to train mode
text_encoder.to(device)
video_encoder.to(device)
video_generator.to(device)
video_discriminator.to(device)

text_encoder.train()
video_encoder.train()
video_generator.train()
video_discriminator.train()

reconstruction_loss_weight=100.0
alignment_loss_weight=10.0

# Define loss functions here to ensure they are in scope
adversarial_loss_fn = nn.BCEWithLogitsLoss()
reconstruction_loss_fn = nn.L1Loss()
def text_video_alignment_loss(text_embeddings, real_video_embeddings):
    text_embeddings_norm = F.normalize(text_embeddings, p=2, dim=1)
    real_video_embeddings_norm = F.normalize(real_video_embeddings, p=2, dim=1)
    similarity = (text_embeddings_norm * real_video_embeddings_norm).sum(dim=1)
    loss = (1 - similarity).mean()
    return loss

# 1. Set a reasonable number of num_epochs for actual training
# For a true text-to-video model, this would be hundreds or thousands.
# For this demonstration, we'll keep it low to avoid very long execution.
num_epochs_actual_training = 5

# 2. Define a directory and file names for saving the trained models
model_save_dir = "./trained_models"
os.makedirs(model_save_dir, exist_ok=True)

generator_model_path = os.path.join(model_save_dir, "video_generator.pth")
discriminator_model_path = os.path.join(model_save_dir, "video_discriminator.pth")

print(f"Training for {num_epochs_actual_training} epochs...")

# 3. Call the train_model function
train_model(
    num_epochs=num_epochs_actual_training,
    train_dataloader=train_dataloader,
    text_encoder=text_encoder,
    video_encoder=video_encoder,
    video_generator=video_generator,
    video_discriminator=video_discriminator,
    optimizer_G=optimizer_G,
    optimizer_D=optimizer_D,
    adversarial_loss_fn=adversarial_loss_fn,
    reconstruction_loss_fn=reconstruction_loss_fn,
    text_video_alignment_loss_fn=text_video_alignment_loss,
    device=device,
    reconstruction_loss_weight=reconstruction_loss_weight,
    alignment_loss_weight=alignment_loss_weight,
    latent_dim=latent_dim
)

# 4. Save the state dictionary of the video_generator
torch.save(video_generator.state_dict(), generator_model_path)
print(f"Trained VideoGenerator model saved to: {generator_model_path}")

# 5. Save the state dictionary of the video_discriminator
torch.save(video_discriminator.state_dict(), discriminator_model_path)
print(f"Trained VideoDiscriminator model saved to: {discriminator_model_path}")

print("Training process completed and models saved.")
# 1. Load the trained video_generator model
# Ensure the model is instantiated with the same architecture as during training

# Instantiate the generator with the same parameters used for training
# These global parameters were defined in previous steps:
# embedding_dim, latent_dim, output_channels, num_frames_per_video, frame_height, frame_width
video_generator_inference = VideoGenerator(
    text_embedding_dim=embedding_dim,
    latent_dim=latent_dim,
    output_channels=output_channels,
    num_frames=num_frames_per_video,
    frame_height=frame_height,
    frame_width=frame_width
).to(device)

# Load the saved state dictionary
# Using map_location to ensure it loads correctly regardless of where it was saved (e.g., GPU saved, CPU loaded)
video_generator_inference.load_state_dict(torch.load(generator_model_path, map_location=device))

# 2. Set the model to evaluation mode
video_generator_inference.eval()

print(f"Loaded VideoGenerator model from: {generator_model_path}")
print("VideoGenerator set to evaluation mode.")
# 3. Define new, unseen text prompts for video generation
new_text_prompts = [
    "A bird flying over a calm lake.",
    "A person walking through a bustling city.",
    "A car racing on a dirt track."
]

# Ensure text_encoder is in evaluation mode
text_encoder.eval()

# 4. Tokenize and encode these new text prompts
# Re-using the tokenizer and text_encoder from previous steps
encoded_new_texts = tokenizer(
    new_text_prompts,
    padding="max_length",
    truncation=True,
    max_length=max_text_length,
    return_tensors="pt"
)

new_input_ids = encoded_new_texts['input_ids'].to(device)
new_attention_mask = encoded_new_texts['attention_mask'].to(device)

# Generate text embeddings for the new prompts
with torch.no_grad():
    new_text_embeddings = text_encoder(new_input_ids, new_attention_mask)

# 5. Generate random latent noise vectors
batch_size_inference = len(new_text_prompts)
new_noise_vector = torch.randn(batch_size_inference, latent_dim).to(device)

print(f"New text prompts: {new_text_prompts}")
print(f"Shape of new text embeddings: {new_text_embeddings.shape}")
print(f"Shape of new noise vectors: {new_noise_vector.shape}")

# 6. Use the video_generator to generate new video frames
with torch.no_grad():
    generated_videos = video_generator_inference(new_text_embeddings, new_noise_vector)

print(f"Shape of generated videos: {generated_videos.shape}")
print("Generated video frames for new prompts successfully.")

# 7. Post-process the generated video frames:
# Move them back to CPU and convert them to NumPy arrays.
# Permute the dimensions from (B, C, D, H, W) to (B, D, H, W, C) for visualization.
processed_videos_np = generated_videos.cpu().numpy().transpose(0, 2, 3, 4, 1)

# De-normalization: reverse the transforms.Normalize applied during data preprocessing
# The normalization values were: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
# Reverse: (pixel * std) + mean
mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 1, 3)
std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 1, 3)

processed_videos_denorm = processed_videos_np * std + mean

# Clip pixel values to the valid range (0-1) and then convert to uint8 (0-255)
processed_videos_uint8 = (processed_videos_denorm.clip(0, 1) * 255).astype(np.uint8)

print(f"Shape of post-processed videos for visualization: {processed_videos_uint8.shape}")

# Create a directory to save generated videos
os.makedirs("generated_videos", exist_ok=True)

# 8. For each generated video, save it as an MP4 file and display it in the Colab output.
print("\n--- Visualizing Generated Videos ---")
for i, video_frames in enumerate(processed_videos_uint8):
    # video_frames shape: (num_frames, H, W, C)
    prompt = new_text_prompts[i]
    output_filename = f"generated_videos/video_{i+1}.mp4"

    # Create a moviepy clip from the sequence of image frames
    clip = ImageSequenceClip(list(video_frames), fps=num_frames_per_video/2) # Adjust FPS as needed
    clip.write_videofile(output_filename, codec="libx264", audio_codec="aac", verbose=False, logger=None)

    print(f"\nGenerated video for prompt: '{prompt}'")
    print(f"Saved to: {output_filename}")

    # 9. Display the video with a legend/caption
    HTML(f"<b>Prompt: '{prompt}'</b>")
    HTML(f"<video width='{frame_width}' controls><source src='{output_filename}' type='video/mp4'></video>")

print("Video generation and visualization complete.")