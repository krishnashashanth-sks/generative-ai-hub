import torch
import torch.nn as nn
from layers import *

class TextToAudioModel(nn.Module):
  def __init__(self,vocab_size,embed_dim,encoder_hidden_size,encoder_num_layers,encoder_dropout_prob,encoder_bidirectional,attention_dim,decoder_hidden_size,n_mels,max_mel_frames,teacher_forcing_ratio=1.0):
    super().__init__()
    self.n_mels=n_mels
    self.max_mel_frames=max_mel_frames
    self.teacher_forcing_ratio=teacher_forcing_ratio
    self.text_encoder=TextEncoder(
        num_embeddings=vocab_size,
        embedding_dim=embed_dim,
        hidden_size=encoder_hidden_size,
        num_layers=encoder_num_layers,
        dropout_prob=encoder_dropout_prob,
        bidirectional=encoder_bidirectional
    )
    encoder_output_dim=encoder_hidden_size*(2 if encoder_bidirectional else 1)
    self.attention_mechanism = AttentionMechanism(
            encoder_output_dim=encoder_output_dim,
            decoder_hidden_dim=decoder_hidden_size, # Assuming decoder hidden size is its own dim
            attention_dim=attention_dim
        )
    decoder_input_dim = encoder_output_dim + n_mels
    self.audio_decoder = AudioDecoder(
            input_dim=decoder_input_dim,
            hidden_size=decoder_hidden_size,
            num_layers=1, # LSTMCell is always a single layer
            output_dim=n_mels
        )
  def forward(self,text_sequences,input_lengths,target_mel_spectrograms=None):
        batch_size=text_sequences.size(0)
        encoder_outputs=self.text_encoder(text_sequences,input_lengths)
        decoder_hidden=torch.zeros(batch_size,self.audio_decoder.hidden_size,device=text_sequences.device)
        decoder_cell=torch.zeros(batch_size,self.audio_decoder.hidden_size,device=text_sequences.device)
        prev_mel_frame=torch.zeros(batch_size,self.n_mels,device=text_sequences.device)
        predicted_mel_frames=[]
        for t in range(self.max_mel_frames):
          context_vector,attention_weights=self.attention_mechanism(encoder_outputs,decoder_hidden)
          predicted_mel_frame,decoder_hidden,decoder_cell=self.audio_decoder(
              context_vector,prev_mel_frame,decoder_hidden,decoder_cell
          )
          predicted_mel_frames.append(predicted_mel_frame)
          use_teacher_forcing=True if target_mel_spectrograms is not None and torch.rand(1).item()<self.teacher_forcing_ratio else False
          if use_teacher_forcing and t<target_mel_spectrograms.size(1)-1:
            prev_mel_frame=target_mel_spectrograms[:,t,:]
          else:
            prev_mel_frame=predicted_mel_frame
        return torch.stack(predicted_mel_frames,dim=1)