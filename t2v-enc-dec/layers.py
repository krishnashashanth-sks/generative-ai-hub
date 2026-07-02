import torch.nn as nn
import torch

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)

    def forward(self, text_sequence):
        text_sequence = text_sequence.long()
        embedded = self.embedding(text_sequence)
        _, hidden = self.gru(embedded)
        return hidden.squeeze(0)

class VideoEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim, num_frames, frame_height, frame_width):
        super(VideoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU()
        )
        with torch.no_grad():
            # Dummy input to calculate flattened size dynamically, using parameters from the previous step
            dummy_input_for_flattened_size = torch.zeros(1, in_channels, num_frames, frame_height, frame_width)
            output_shape = self.conv_layers(dummy_input_for_flattened_size).shape
            flattened_size = output_shape[1] * output_shape[2] * output_shape[3] * output_shape[4]
        self.fc = nn.Linear(flattened_size, latent_dim)

    def forward(self, video_sequence):
        video_sequence = video_sequence.permute(0, 4, 1, 2, 3)
        x = self.conv_layers(video_sequence)
        x = x.reshape(x.size(0), -1)
        latent_vector = self.fc(x)
        return latent_vector

class VideoDecoder(nn.Module):
    def __init__(self, latent_dim, out_channels, num_frames, frame_height, frame_width):
        super(VideoDecoder, self).__init__()
        self.out_channels = out_channels
        self.num_frames = num_frames
        self.frame_height = frame_height
        self.frame_width = frame_width

        self.initial_channels = 512 # Doubled from 256 in previous architecture
        self.z_D = 13
        self.z_H = 8
        self.z_W = 8

        self.fc = nn.Linear(latent_dim, self.initial_channels * self.z_D * self.z_H * self.z_W)

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose3d(self.initial_channels, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.ConvTranspose3d(256, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.ConvTranspose3d(128, out_channels, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1)),
            nn.Tanh()
        )

    def forward(self, latent_vector):
        x = self.fc(latent_vector)
        x = x.view(latent_vector.size(0), self.initial_channels, self.z_D, self.z_H, self.z_W)
        x = self.deconv_layers(x)
        x = x.permute(0, 2, 3, 4, 1)
        return x
