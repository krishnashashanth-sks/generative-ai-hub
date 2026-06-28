from dataset import train_dataset,test_dataset,val_dataset,vocabulary,N_MELS,train_vocoder_dataset,val_vocoder_dataset,EOS_ID,TARGET_DIR,TARGET_SR,char_to_id,SOS_ID,UNK_ID
from torch.utils.data import DataLoader
import torch
from acoustic_model import FastSpeech2
from vocoder_model import HiFiGAN
import torch.nn as nn
from utils import tts_collate_fn,vocoder_collate_fn
from train import train_acoustic_model,train_vocoder_model
import torch.optim as optim
import os
import soundfile as sf
from inference import synthesize_speech_end_to_end

BATCH_SIZE_ACOUSTIC = 32
LEARNING_RATE_ACOUSTIC = 1e-4
NUM_EPOCHS_ACOUSTIC = 50

# Create DataLoader instances
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE_ACOUSTIC, shuffle=True, collate_fn=tts_collate_fn, num_workers=2
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE_ACOUSTIC, shuffle=False, collate_fn=tts_collate_fn, num_workers=2
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE_ACOUSTIC, shuffle=False, collate_fn=tts_collate_fn, num_workers=2
)

# Encoder Hyperparameters (from `ece8e7ce`)
ENCODER_D_MODEL = 256
ENCODER_N_HEADS = 4
ENCODER_D_FF = 1024
ENCODER_N_LAYERS = 6
ENCODER_DROPOUT_RATE = 0.1
# Use the dynamically determined max length from the kernel state explicitly
MAX_PHONEME_SEQ_LEN = 518 # This value was determined by the previous step's analysis

# Variance Adaptor Hyperparameters (from `ece8e7ce`)
FILTER_SIZE = 256
KERNEL_SIZE = 3
DROPOUT = 0.1 # General dropout rate used for Variance Adaptor components
PITCH_EMBEDDING_DIM = ENCODER_D_MODEL
ENERGY_EMBEDDING_DIM = ENCODER_D_MODEL
NUM_PITCH_BINS = 256 # Example number of bins
NUM_ENERGY_BINS = 256 # Example number of bins

# Decoder Hyperparameters (from `ece8e7ce`)
DECODER_D_MODEL = ENCODER_D_MODEL
DECODER_N_HEADS = 4
DECODER_D_FF = 1024
DECODER_N_LAYERS = 6
DECODER_DROPOUT_RATE = 0.1
MAX_MEL_SEQ_LEN = 1000 # Max expected length of generated Mel spectrogram
VOCAB_SIZE = len(vocabulary)

acoustic_model = FastSpeech2(
    vocab_size=VOCAB_SIZE,
    encoder_d_model=ENCODER_D_MODEL, encoder_n_heads=ENCODER_N_HEADS, encoder_d_ff=ENCODER_D_FF,
    encoder_n_layers=ENCODER_N_LAYERS, encoder_dropout_rate=ENCODER_DROPOUT_RATE,
    max_phoneme_seq_len=MAX_PHONEME_SEQ_LEN,
    variance_adaptor_filter_size=FILTER_SIZE, variance_adaptor_kernel_size=KERNEL_SIZE,
    variance_adaptor_dropout=DROPOUT,
    pitch_embedding_dim=PITCH_EMBEDDING_DIM, energy_embedding_dim=ENERGY_EMBEDDING_DIM,
    num_pitch_bins=NUM_PITCH_BINS, num_energy_bins=NUM_ENERGY_BINS,
    decoder_d_model=DECODER_D_MODEL, decoder_n_heads=DECODER_N_HEADS, decoder_d_ff=DECODER_D_FF,
    decoder_n_layers=DECODER_N_LAYERS, decoder_dropout_rate=DECODER_DROPOUT_RATE,
    n_mels=N_MELS, max_mel_seq_len=MAX_MEL_SEQ_LEN
)

mel_criterion = nn.L1Loss() # L1 Loss is common for Mel spectrograms
duration_criterion = nn.MSELoss() # MSE Loss for duration prediction
pitch_criterion = nn.MSELoss() # MSE Loss for pitch prediction
energy_criterion = nn.MSELoss() # MSE Loss for energy prediction

optimizer = optim.Adam(acoustic_model.parameters(), lr=LEARNING_RATE_ACOUSTIC)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
acoustic_model.to(device)

train_acoustic_model(NUM_EPOCHS_ACOUSTIC,acoustic_model,train_loader,val_loader,optimizer,mel_criterion,duration_criterion,pitch_criterion,energy_criterion,device)

BATCH_SIZE_VOCODER = 4 # Adjust based on GPU memory. Started with 1, increased slightly.
LEARNING_RATE_VOCODER = 1e-4
NUM_EPOCHS_VOCODER = 20

train_vocoder_loader = DataLoader(
    train_vocoder_dataset, batch_size=BATCH_SIZE_VOCODER, shuffle=True, collate_fn=vocoder_collate_fn, num_workers=2, pin_memory=True
)
val_vocoder_loader = DataLoader(
    val_vocoder_dataset, batch_size=BATCH_SIZE_VOCODER, shuffle=False, collate_fn=vocoder_collate_fn, num_workers=2, pin_memory=True
)

vocoder_model = HiFiGAN(n_mels=N_MELS)

criterion_vocoder = nn.L1Loss() # Mean Absolute Error
optimizer_vocoder = optim.Adam(vocoder_model.parameters(), lr=LEARNING_RATE_VOCODER)

vocoder_model.to(device)

train_vocoder_model(NUM_EPOCHS_VOCODER,vocoder_model,train_vocoder_loader,val_vocoder_loader,optimizer_vocoder,criterion_vocoder,device)

OUTPUT_AUDIO_DIR = os.path.join(TARGET_DIR, 'generated_audio')
os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

sample_text_inference = "The quick brown fox jumps over the lazy dog."

generated_audio_inference = synthesize_speech_end_to_end(
    sample_text_inference, acoustic_model, vocoder_model, char_to_id, SOS_ID, EOS_ID, UNK_ID, device
)

output_filepath_inference = os.path.join(OUTPUT_AUDIO_DIR, 'generated_inference_sample.wav')
sf.write(output_filepath_inference, generated_audio_inference, TARGET_SR)