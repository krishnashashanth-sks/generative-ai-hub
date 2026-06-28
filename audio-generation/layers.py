import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from dataset import PAD_ID,SEGMENT_SAMPLES,N_MELS

class MultiHeadAttention(nn.Module):
  def __init__(self,d_model,n_heads):
    super(MultiHeadAttention,self).__init__()
    self.d_model=d_model
    self.n_heads=n_heads
    self.head_dim=d_model//n_heads
    self.q_linear=nn.Linear(d_model,d_model)
    self.k_linear=nn.Linear(d_model,d_model)
    self.v_linear=nn.Linear(d_model,d_model)
    self.out=nn.Linear(d_model,d_model)
  def forward(self,q,k,v,mask=None):
    batch_size=q.size(0)
    q=self.q_linear(q).view(batch_size,-1,self.n_heads,self.head_dim)
    k=self.k_linear(k).view(batch_size,-1,self.n_heads,self.head_dim)
    v=self.v_linear(v).view(batch_size,-1,self.n_heads,self.head_dim)
    q=q.transpose(1,2)
    k=k.transpose(1,2)
    v=v.transpose(1,2)
    scores=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.head_dim)
    if mask is not None:
      scores=scores.masked_fill(mask==0,-1e9)
    attention=F.softmax(scores,dim=-1)
    concat=torch.matmul(attention,v)
    concat=concat.transpose(1,2).contiguous().view(batch_size,-1,self.d_model)
    return self.out(concat)

class FeedForward(nn.Module):
  def __init__(self,d_model,d_ff,dropout_rate=0.1):
    super(FeedForward,self).__init__()
    self.linear1=nn.Linear(d_model,d_ff)
    self.dropout=nn.Dropout(dropout_rate)
    self.linear2=nn.Linear(d_ff,d_model)
  def forward(self,x):
    return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
  def __init__(self,d_model,n_heads,d_ff,dropout_rate=0.1):
    super(EncoderLayer,self).__init__()
    self.attn=MultiHeadAttention(d_model,n_heads)
    self.feed_forward=FeedForward(d_model,d_ff,dropout_rate)
    self.norm1=nn.LayerNorm(d_model)
    self.norm2=nn.LayerNorm(d_model)
    self.dropout1=nn.Dropout(dropout_rate)
    self.dropout2=nn.Dropout(dropout_rate)
  def forward(self,x,mask=None):
    attn_output=self.attn(x,x,x,mask)
    x=self.norm1(x+self.dropout1(attn_output))
    ff_output=self.feed_forward(x)
    x=self.norm2(x+self.dropout2(ff_output))
    return x

class FastSpeech2Encoder(nn.Module):
  def __init__(self,vocab_size,d_model,n_heads,d_ff,n_layers,max_seq_len,dropout_rate=0.1):
    super(FastSpeech2Encoder,self).__init__()
    self.d_model=d_model
    self.phoneme_embedding=nn.Embedding(vocab_size,d_model,padding_idx=PAD_ID)
    self.positional_encoding=nn.Parameter(self._get_sinusoid_encoding_table(max_seq_len,d_model),requires_grad=False)
    self.encoder_layers=nn.ModuleList([
        EncoderLayer(d_model,n_heads,d_ff,dropout_rate)
        for _ in range(n_layers)
    ])
    self.dropout=nn.Dropout(dropout_rate)
    self.norm=nn.LayerNorm(d_model)
  def _get_sinusoid_encoding_table(self,n_position,d_hid):
    def get_position_angle_vec(position):
      return [position/np.power(10000,2*(hid_j//2)/d_hid) for hid_j in range(d_hid)]
    sinusoid_table=np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:,0::2]=np.sin(sinusoid_table[:,0::2])
    sinusoid_table[:,1::2]=np.cos(sinusoid_table[:,1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)
  def forward(self,phoneme_sequence,mask=None):
    seq_len=phoneme_sequence.size(1)
    x=self.phoneme_embedding(phoneme_sequence)

    pos_enc_max_len = self.positional_encoding.shape[1]
    if seq_len > pos_enc_max_len:
        print(f"Warning: Current batch sequence length ({seq_len}) exceeds positional encoding max length ({pos_enc_max_len}). Truncating phoneme_sequence for positional encoding addition.")
        # Truncate phoneme_sequence and x to fit positional encoding length
        x = x[:, :pos_enc_max_len, :]
        seq_len = pos_enc_max_len
        # Also adjust mask if present to match truncated x
        if mask is not None:
             mask = mask[:, :pos_enc_max_len]

    # Perform addition after potential truncation
    sliced_positional_encoding = self.positional_encoding[:,:seq_len,:]
    
    seq_len = x.size(1)
    x=x+self.positional_encoding[:,:seq_len,:]

    x=self.dropout(x)
    if mask is not None:
        # Ensure attention_mask also matches the potentially truncated seq_len
        attention_mask = mask.unsqueeze(1).unsqueeze(2)[:, :, :, :seq_len] # (batch_size, 1, 1, seq_len)
    else:
        attention_mask = None
    for layer in self.encoder_layers:
      x=layer(x,attention_mask)
    x=self.norm(x)
    return x

class ConvBlock(nn.Module):
  def __init__(self,in_channels,out_channels,kernel_size,padding,dropout_rate):
    super(ConvBlock,self).__init__()
    self.conv=nn.Conv1d(in_channels,out_channels,kernel_size=kernel_size,padding=padding)
    self.relu=nn.ReLU()
    self.norm=nn.LayerNorm(out_channels)
    self.dropout=nn.Dropout(dropout_rate)
  def forward(self,x,mask=None):
    x=self.conv(x)
    x=self.relu(x)
    x=x.permute(0,2,1)
    x=self.norm(x)
    x=x.permute(0,2,1)
    x=self.dropout(x)
    if mask is not None:
      x = x.masked_fill(mask == 0, 0.0)
    return x

class VariancePredictor(nn.Module):
  def __init__(self,d_model,filter_size,kernel_size,dropout_rate):
    super(VariancePredictor,self).__init__()
    self.conv_blocks=nn.ModuleList([
        ConvBlock(d_model,filter_size,kernel_size,padding=(kernel_size-1)//2,dropout_rate=dropout_rate),
        ConvBlock(filter_size,d_model,kernel_size,padding=(kernel_size-1)//2,dropout_rate=dropout_rate)
    ])
    self.linear=nn.Linear(d_model,1)
  def forward(self,x,mask=None):
    x=x.permute(0,2,1)
    conv_mask = mask.unsqueeze(1) if mask is not None else None
    for conv_block in self.conv_blocks:
      x=conv_block(x,conv_mask)
    x=x.permute(0,2,1)
    prediction=self.linear(x).squeeze(-1)
    if mask is not None:
      prediction=prediction.masked_fill(mask == 0, 0.0)
    return prediction
  
class VarianceAdaptor(nn.Module):
  def __init__(self,encoder_dim,filter_size,kernel_size,dropout,pitch_embedding_dim,energy_embedding_dim,num_pitch_bins,num_energy_bins):
    super(VarianceAdaptor,self).__init__()
    self.duration_predictor=VariancePredictor(encoder_dim,filter_size,kernel_size,dropout)
    self.pitch_predictor=VariancePredictor(encoder_dim,filter_size,kernel_size,dropout)
    self.energy_predictor=VariancePredictor(encoder_dim,filter_size,kernel_size,dropout)
    self.pitch_embedding=nn.Embedding(num_pitch_bins,pitch_embedding_dim)
    self.energy_embedding=nn.Embedding(num_energy_bins,energy_embedding_dim)

  def _length_regulate(self, encoder_output, durations, mask):
      batch_size, seq_len, encoder_dim = encoder_output.shape

      masked_durations = durations * mask.long()
      max_output_len = torch.max(torch.sum(masked_durations, dim=-1)).item()

      output = torch.zeros(batch_size, int(max_output_len), encoder_dim, device=encoder_output.device)
      expanded_mask = torch.zeros(batch_size, int(max_output_len), dtype=torch.bool, device=encoder_output.device)

      for i in range(batch_size):
          pos = 0
          for j in range(seq_len):
              if mask[i, j]:
                  dur = int(durations[i, j].item())
                  if dur > 0:
                      if pos + dur > max_output_len:
                          dur = int(max_output_len - pos)
                          if dur <= 0:
                              break
                      output[i, pos : pos + dur, :] = encoder_output[i, j, :].unsqueeze(0).repeat(dur, 1)
                      expanded_mask[i, pos : pos + dur] = True
                      pos += dur
      return output, expanded_mask.unsqueeze(1)

  def forward(self, encoder_output, mask, durations_target=None, pitch_target=None, energy_target=None):
    predicted_durations = self.duration_predictor(encoder_output, mask)
    predicted_pitch = self.pitch_predictor(encoder_output, mask)
    predicted_energy = self.energy_predictor(encoder_output, mask)

    if durations_target is not None:
      durations = durations_target
    else:
      durations = torch.clamp(torch.round(torch.exp(predicted_durations) - 1), min=0).long()
      durations = durations * mask.long()

    if pitch_target is not None:
      pitch = pitch_target
    else:
      pitch = torch.round(predicted_pitch * (self.pitch_embedding.num_embeddings - 1)).long()
      pitch = torch.clamp(pitch, 0, self.pitch_embedding.num_embeddings - 1)
      pitch = pitch * mask.long()

    if energy_target is not None:
      energy = energy_target
    else:
      energy = torch.round(predicted_energy * (self.energy_embedding.num_embeddings - 1)).long()
      energy = torch.clamp(energy, 0, self.energy_embedding.num_embeddings - 1)
      energy = energy * mask.long()

    pitch_embedded = self.pitch_embedding(pitch)
    energy_embedded = self.energy_embedding(energy)

    encoder_output_conditioned = encoder_output + pitch_embedded + energy_embedded

    output_expanded, expanded_mask = self._length_regulate(encoder_output_conditioned, durations, mask)

    return output_expanded, predicted_durations, predicted_pitch, predicted_energy, expanded_mask

class MelSpectrogramDecoder(nn.Module):
  def __init__(self,d_model,n_heads,d_ff,n_layers,dropout_rate,n_mels,max_expanded_len=1000):
    super(MelSpectrogramDecoder,self).__init__()
    self.d_model=d_model
    self.n_mels=n_mels
    self.positional_encoding=nn.Parameter(
        self._get_sinusoid_encoding_table(max_expanded_len,d_model),
        requires_grad=False
    )
    self.decoder_layers=nn.ModuleList([
        EncoderLayer(d_model,n_heads,d_ff,dropout_rate)
        for _ in range(n_layers)
    ])
    self.dropout=nn.Dropout(dropout_rate)
    self.layer_norm=nn.LayerNorm(d_model)
    self.linear_projection=nn.Linear(d_model,n_mels)

  def _get_sinusoid_encoding_table(self,n_position,d_hid):
    def get_position_angle_vec(position):
      return [position/np.power(10000,2*(hid_j//2)/d_hid) for hid_j in range(d_hid)]
    sinusoid_table=np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:,0::2]=np.sin(sinusoid_table[:,0::2])
    sinusoid_table[:,1::2]=np.cos(sinusoid_table[:,1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

  def forward(self,expanded_encoder_output,expanded_mask):
    expanded_seq_len=expanded_encoder_output.size(1)
    if expanded_seq_len > self.positional_encoding.shape[1]:
        expanded_encoder_output = expanded_encoder_output[:, :self.positional_encoding.shape[1], :]
        expanded_mask = expanded_mask[:, :, :self.positional_encoding.shape[1]]
        expanded_seq_len = expanded_encoder_output.size(1)

    x=expanded_encoder_output+self.positional_encoding[:,:expanded_seq_len,:]
    x=self.dropout(x)

    attention_mask = expanded_mask.unsqueeze(2)

    for layer in self.decoder_layers:
      x=layer(x,attention_mask)

    x=self.layer_norm(x)
    mel_output=self.linear_projection(x)
    return mel_output.transpose(1,2)
  
class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock, self).__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=(kernel_size - 1) * d // 2),
                nn.LeakyReLU(0.2),
                nn.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=(kernel_size - 1) * d // 2),
                nn.LeakyReLU(0.2)
            ) for d in dilation
        ])

    def forward(self, x):
        for conv_seq in self.convs:
            x = x + conv_seq(x)
        return x

class HiFiGANGenerator(nn.Module):
    def __init__(self, n_mels=N_MELS, num_upsamples=(8, 8, 2, 2), upsample_kernel_sizes=(16, 16, 4, 4),
                 upsample_initial_channel=512, resblock_kernel_sizes=(3, 7, 11), resblock_dilations=([1, 3, 5], [1, 3, 5], [1, 3, 5])):
        super(HiFiGANGenerator, self).__init__()
        self.num_upsamples = len(num_upsamples)
        self.upsample_initial_channel = upsample_initial_channel
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilations = resblock_dilations

        self.input_conv = nn.Conv1d(n_mels, upsample_initial_channel, 7, 1, padding=3)

        self.upsamples = nn.ModuleList()
        for i, (u, k) in enumerate(zip(num_upsamples, upsample_kernel_sizes)):
            self.upsamples.append(nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.ConvTranspose1d(upsample_initial_channel // (2**i), upsample_initial_channel // (2**(i+1)),
                                   k, u, padding=(k-u)//2)
            ))

        self.resblocks = nn.ModuleList()
        for i in range(self.num_upsamples):
            ch = upsample_initial_channel // (2**(i+1))
            if i < len(self.resblock_dilations):
                current_dilations = self.resblock_dilations[i]
            else:
                current_dilations = self.resblock_dilations[-1]

            for j, k in enumerate(self.resblock_kernel_sizes):
                self.resblocks.append(ResBlock(ch, k, current_dilations))

        self.output_conv = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv1d(ch, 1, 7, 1, padding=3),
            nn.Tanh()
        )

    def forward(self, mel_spectrogram):
        if mel_spectrogram.dim() == 2:
            mel_spectrogram = mel_spectrogram.unsqueeze(0)

        x = self.input_conv(mel_spectrogram)

        for i in range(self.num_upsamples):
            x = self.upsamples[i](x)
            start_idx = i * len(self.resblock_kernel_sizes)
            end_idx = start_idx + len(self.resblock_kernel_sizes)
            for j in range(start_idx, min(end_idx, len(self.resblocks))):
                x = self.resblocks[j](x)

        output = self.output_conv(x)

        if output.shape[-1] > SEGMENT_SAMPLES:
            output = output[..., :SEGMENT_SAMPLES]
        elif output.shape[-1] < SEGMENT_SAMPLES:
            pad_size = SEGMENT_SAMPLES - output.shape[-1]
            output = F.pad(output, (0, pad_size), 'constant', 0)

        return output.squeeze(1)
