import torch.nn as nn
from layers import *
import torch

class VideoTextGenerator(nn.Module):
    def __init__(self, vocab_size, text_embedding_dim, text_hidden_size,
                 video_in_channels, video_latent_dim, num_frames, frame_height, frame_width):
        super(VideoTextGenerator, self).__init__()

        self.text_encoder = TextEncoder(vocab_size, text_embedding_dim, text_hidden_size)
        self.video_encoder = VideoEncoder(video_in_channels, video_latent_dim, num_frames, frame_height, frame_width)

        self.combined_latent_dim = text_hidden_size + video_latent_dim

        self.video_decoder = VideoDecoder(self.combined_latent_dim, video_in_channels, num_frames, frame_height, frame_width)

    def forward(self, video_sequence, text_sequence):
        video_latent = self.video_encoder(video_sequence)
        text_embedding = self.text_encoder(text_sequence)
        combined_latent = torch.cat((video_latent, text_embedding), dim=1)
        generated_video = self.video_decoder(combined_latent)
        return generated_video