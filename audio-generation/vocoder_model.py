import torch.nn as nn
from layers import HiFiGANGenerator
from dataset import N_MELS

class HiFiGAN(nn.Module):
    def __init__(self, n_mels=N_MELS):
        super(HiFiGAN, self).__init__()
        self.generator = HiFiGANGenerator(n_mels=n_mels)
        print("HiFiGAN model (with detailed generator) initialized.")

    def forward(self, mel_spectrogram):
        return self.generator(mel_spectrogram)
