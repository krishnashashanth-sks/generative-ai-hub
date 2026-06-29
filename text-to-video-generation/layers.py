import torch.nn as nn
import torch.nn.functional as F

class TextEncoder(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=2):
    super(TextEncoder, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.gru = nn.GRU(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True)

  def forward(self, text_sequence):
    embedded = self.embedding(text_sequence)
    _, hidden = self.gru(embedded)
    return hidden[-1] # Return the hidden state of the last layer

class VideoGenerator(nn.Module):
    def __init__(self, text_embedding_dim, video_channels, video_frames, video_height, video_width):
        super(VideoGenerator, self).__init__()
        self.video_frames = video_frames
        self.video_height = video_height
        self.video_width = video_width

        # Calculate initial latent dimensions, assuming 8x downsampling in each dimension
        initial_depth = self.video_frames // 8
        initial_height = self.video_height // 8
        initial_width = self.video_width // 8
        self.initial_channels = 128 # Base channels for the latent space

        # Fully connected layer to transform text embedding to initial 3D latent space
        self.fc = nn.Linear(text_embedding_dim, self.initial_channels * initial_depth * initial_height * initial_width)

        self.deconv_layers = nn.Sequential(
            # Layer 1: Upsample depth, height, width by 2
            nn.ConvTranspose3d(self.initial_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            # Layer 2: Upsample depth, height, width by 2
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            # Layer 3: Upsample depth, height, width by 2
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            # Layer 4: Output layer, matching video_channels
            nn.ConvTranspose3d(64, video_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Tanh for output pixel values between -1 and 1
        )

        # Store initial latent dimensions for reshape
        self.initial_depth = initial_depth
        self.initial_height = initial_height
        self.initial_width = initial_width

    def forward(self, text_embedding):
        x = self.fc(text_embedding)

        # Reshape to 5D tensor for ConvTranspose3d: (batch_size, channels, depth, height, width)
        x = x.view(-1, self.initial_channels, self.initial_depth, self.initial_height, self.initial_width)

        video = self.deconv_layers(x)

        # Ensure exact output dimensions using interpolate if needed (e.g., due to rounding from //8)
        if video.shape[2] != self.video_frames or video.shape[3] != self.video_height or video.shape[4] != self.video_width:
             video = F.interpolate(video, size=(self.video_frames, self.video_height, self.video_width), mode='trilinear', align_corners=False)

        return video