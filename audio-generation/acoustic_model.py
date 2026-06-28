import torch.nn as nn
from layers import *

class FastSpeech2(nn.Module):
    def __init__(self, vocab_size, encoder_d_model, encoder_n_heads, encoder_d_ff,
                 encoder_n_layers, encoder_dropout_rate, max_phoneme_seq_len,
                 variance_adaptor_filter_size, variance_adaptor_kernel_size,
                 variance_adaptor_dropout, pitch_embedding_dim, energy_embedding_dim,
                 num_pitch_bins, num_energy_bins, decoder_d_model, decoder_n_heads,
                 decoder_d_ff, decoder_n_layers, decoder_dropout_rate, n_mels, max_mel_seq_len):
        super(FastSpeech2, self).__init__()
        self.encoder = FastSpeech2Encoder(
            vocab_size=vocab_size,
            d_model=encoder_d_model,
            n_heads=encoder_n_heads,
            d_ff=encoder_d_ff,
            n_layers=encoder_n_layers,
            dropout_rate=encoder_dropout_rate,
            max_seq_len=max_phoneme_seq_len # Use the updated max_phoneme_seq_len
        )
        self.variance_adaptor = VarianceAdaptor(
            encoder_dim=encoder_d_model,
            filter_size=variance_adaptor_filter_size,
            kernel_size=variance_adaptor_kernel_size,
            dropout=variance_adaptor_dropout,
            pitch_embedding_dim=pitch_embedding_dim,
            energy_embedding_dim=energy_embedding_dim,
            num_pitch_bins=num_pitch_bins,
            num_energy_bins=num_energy_bins
        )
        self.decoder = MelSpectrogramDecoder(
            d_model=decoder_d_model,
            n_heads=decoder_n_heads,
            d_ff=decoder_d_ff,
            n_layers=decoder_n_layers,
            dropout_rate=decoder_dropout_rate,
            n_mels=n_mels,
            max_expanded_len=max_mel_seq_len
        )

    def forward(self, phoneme_sequence, phoneme_mask, duration_target=None, pitch_target=None, energy_target=None):
        encoder_output = self.encoder(phoneme_sequence, mask=phoneme_mask)

        variance_adaptor_output, predicted_durations, predicted_pitch, predicted_energy, expanded_mask = \
            self.variance_adaptor(
                encoder_output, phoneme_mask,
                durations_target=duration_target,
                pitch_target=pitch_target,
                energy_target=energy_target
            )

        mel_output = self.decoder(variance_adaptor_output, expanded_mask)

        return mel_output, predicted_durations, predicted_pitch, predicted_energy
