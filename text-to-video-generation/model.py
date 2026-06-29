import torch.nn as nn
from layers import *

class TextToVideoModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, video_channels, video_frames, video_height, video_width, num_gru_layers=2):
        super(TextToVideoModel, self).__init__()
        self.text_encoder = TextEncoder(vocab_size, embedding_dim, hidden_size, num_layers=num_gru_layers)
        self.video_generator = VideoGenerator(hidden_size, video_channels, video_frames, video_height, video_width)

    def forward(self, text_sequence):
        text_embedding = self.text_encoder(text_sequence)
        video_output = self.video_generator(text_embedding)
        return video_output