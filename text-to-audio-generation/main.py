import torch
import os
import torch.optim as optim
from sklearn.model_selection import train_test_split
from utils import preprocess_text
from dataset import LJSpeechDataset
import pandas as pd
from model import TextToAudioModel
from torch.utils.data import DataLoader
import torch.nn as nn
from train import train_model
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from inference import synthesize_mel_spectrogram

#Use the below link to download the dataset files
# https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2

metadata_path = 'LJSpeech-1.1/metadata.csv'
metadata_df = pd.read_csv(metadata_path, sep='|', header=None)

metadata_df.columns = ['audio_filename', 'transcription', 'normalized_transcription']

metadata_df['preprocessed_text'] = metadata_df['transcription'].apply(preprocess_text)

metadata_df['audio_file_path'] = metadata_df['audio_filename'].apply(lambda x: os.path.join('LJSpeech-1.1/wavs/', x + '.wav'))

# Define a random seed for reproducibility
RANDOM_SEED = 42

# Split the dataset into training and temporary sets (80% train, 20% temp)
train_df, temp_df = train_test_split(metadata_df, test_size=0.2, random_state=RANDOM_SEED)

# Split the temporary set into validation and test sets (10% validation, 10% test from original)
# This means temp_df is split into two halves, each being 10% of the original dataset
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_SEED)

vocabulary_size = 100
embedding_dim = 256
encoder_hidden_size = 512
encoder_num_layers = 2
encoder_dropout_prob = 0.5
encoder_bidirectional = True
attention_dim = 256
decoder_hidden_size = 512
n_mels = 80
max_mel_frames = 500 # A reasonable max length for Mel-spectrograms

model = TextToAudioModel(
    vocab_size=vocabulary_size,
    embed_dim=embedding_dim,
    encoder_hidden_size=encoder_hidden_size,
    encoder_num_layers=encoder_num_layers,
    encoder_dropout_prob=encoder_dropout_prob,
    encoder_bidirectional=encoder_bidirectional,
    attention_dim=attention_dim,
    decoder_hidden_size=decoder_hidden_size,
    n_mels=n_mels,
    max_mel_frames=max_mel_frames
)
loss_fn = nn.MSELoss()


# 1. Define a learning rate
learning_rate = 1e-3  # Common starting learning rate for Adam

# 2. Instantiate the Adam optimizer
# We pass all parameters of our integrated model to the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define a character-to-index mapping (tokenizer)
# Collect all unique characters from the training data for a robust vocabulary
all_chars = sorted(list(set(''.join(train_df['preprocessed_text'].tolist()))))
char_to_idx = {char: i + 1 for i, char in enumerate(all_chars)} # +1 for padding
char_to_idx['<pad>'] = 0 # Add padding token
vocabulary_size = len(char_to_idx)

print(f"Vocabulary size: {vocabulary_size}")
print(f"Example char_to_idx: {list(char_to_idx.items())[:10]}")

def collate_fn(batch):
  """
    Custom collate_fn for padding sequences in a batch.
    Pads text sequences and Mel-spectrograms to the maximum length within the batch.
    """
  text_sequences=[item[0] for item in batch]
  mel_spectrograms=[item[1] for item in batch]
  text_lengths=torch.tensor([len(seq) for seq in text_sequences],dtype=torch.long)
  mel_lengths=torch.tensor([len(mel) for mel in mel_spectrograms],dtype=torch.long)
  padded_text_sequences=pad_sequence(text_sequences,batch_first=True,padding_value=char_to_idx['<pad>'])
  padded_mel_spectrograms=pad_sequence(mel_spectrograms,batch_first=True,padding_value=0.0)
  return padded_text_sequences,text_lengths,padded_mel_spectrograms,mel_lengths

# Instantiate the datasets
# Re-using parameters from previous steps like TARGET_SAMPLE_RATE, n_mels (from model init), etc.
# Assuming n_fft and hop_length are 2048 and 512 respectively as used in audio_to_mel_spectrogram demo
n_mels = 80
n_fft = 2048
hop_length = 512

# Define the target sample rate
TARGET_SAMPLE_RATE = 22050

train_dataset = LJSpeechDataset(train_df, char_to_idx, TARGET_SAMPLE_RATE, n_mels, n_fft, hop_length)
val_dataset = LJSpeechDataset(val_df, char_to_idx, TARGET_SAMPLE_RATE, n_mels, n_fft, hop_length)
test_dataset = LJSpeechDataset(test_df, char_to_idx, TARGET_SAMPLE_RATE, n_mels, n_fft, hop_length)

# Define DataLoader parameters
batch_size = 1# You can adjust this
# Change num_workers to 0 to debug multiprocessing issues
num_workers = 0 # os.cpu_count()# Number of subprocesses to use for data loading

# Create DataLoader instances
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers
)
# Set device for training (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to the device
model.to(device)

# Training parameters
num_epochs = 10  # You can adjust the number of epochs

train_losses,val_losses,best_val_loss,best_model_state=train_model(num_epochs,model,train_loader,val_loader,loss_fn,optimizer,device)
print("Training complete.")

# Load the best model state after training
if best_model_state:
    model.load_state_dict(best_model_state)
    print("Best model state loaded into model.")


plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()


# Sample text for inference
sample_inference_text = "This is a test sentence for text to audio generation."

print(f"\nGenerating Mel-spectrogram for: '{sample_inference_text}'")

# Generate Mel-spectrogram
generated_mel_spec = synthesize_mel_spectrogram(text_to_audio_model, sample_inference_text, char_to_idx, device)

if generated_mel_spec is not None:
    print(f"Shape of generated Mel-spectrogram: {generated_mel_spec.shape}")

    # Visualize the generated Mel-spectrogram
    plt.figure(figsize=(14, 6))
    librosa.display.specshow(generated_mel_spec.T, sr=TARGET_SAMPLE_RATE, x_axis='time', y_axis='mel',
                             hop_length=hop_length, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Generated Mel-spectrogram for: "{sample_inference_text}"')
    plt.tight_layout()
    plt.show()
else:
    print("Mel-spectrogram generation failed.")
