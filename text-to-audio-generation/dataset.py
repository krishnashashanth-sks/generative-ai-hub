import torch
from torch.utils.data import Dataset
from utils import preprocess_audio,audio_to_mel_spectrogram

class LJSpeechDataset(Dataset):
  def __init__(self,dataframe,char_to_idx,target_sr,n_mels,n_fft,hop_length):
    self.dataframe=dataframe
    self.char_to_idx=char_to_idx
    self.target_sr=target_sr
    self.n_mels=n_mels
    self.n_fft=n_fft
    self.hop_length=hop_length
  def __len__(self):
    return len(self.dataframe)
  def __getitem__(self,idx):
    row=self.dataframe.iloc[idx]
    audio_path=row['audio_file_path']
    preprocessed_audio_waveform=preprocess_audio(audio_path,self.target_sr)
    mel_spectrogram=audio_to_mel_spectrogram(preprocessed_audio_waveform,self.target_sr,self.n_mels,self.n_fft,self.hop_length)
    # Transpose here so that the shape becomes (n_frames, n_mels) to allow padding on n_frames
    mel_spectrogram=torch.tensor(mel_spectrogram.T, dtype=torch.float32)
    text=row['preprocessed_text']
    tokenized_text=[self.char_to_idx[char] for char in text if char in self.char_to_idx]
    text_tensor=torch.tensor(tokenized_text,dtype=torch.long)
    return text_tensor,mel_spectrogram