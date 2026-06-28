from sklearn.model_selection import train_test_split
import numpy as np
import urllib.request
import tarfile
import os
from utils import install_package,copy_segments_to_split
import librosa
import soundfile as sf
import glob
import shutil # Added for rmtree
from torch.utils.data import Dataset
import torch
from utils import text_to_sequence

# Define the URL for a sample audio dataset (e.g., LibriSpeech dev-clean subset)
# For demonstration purposes, we will use a small subset of LibriSpeech dev-clean (approx 35MB)
DATASET_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
DATASET_NAME = "dev-clean.tar.gz"
TARGET_DIR = "./audio_dataset"

# Create the target directory if it doesn't exist
os.makedirs(TARGET_DIR, exist_ok=True)

# Define the full path for the downloaded archive
download_path = os.path.join(TARGET_DIR, DATASET_NAME)

print(f"Downloading dataset from {DATASET_URL} to {download_path}...")
# Download the dataset
try:
    urllib.request.urlretrieve(DATASET_URL, download_path)
    print("Download complete.")
except Exception as e:
    print(f"Error during download: {e}")
    print("Please check the URL and your internet connection.")

print(f"Extracting {DATASET_NAME} to {TARGET_DIR}...")
# Extract the dataset
try:
    with tarfile.open(download_path, "r:gz") as tar:
        # Added filter='data' to address DeprecationWarning in Python 3.14+
        tar.extractall(path=TARGET_DIR, filter='data')
    print("Extraction complete.")
except tarfile.ReadError as e:
    print(f"Error during extraction: {e}")
    print("The downloaded file might be corrupted or not a valid tar.gz archive.")
except Exception as e:
    print(f"An unexpected error occurred during extraction: {e}")

# Verify the download and extraction by listing contents
print(f"Listing contents of {TARGET_DIR}:")
if os.path.exists(TARGET_DIR):
    for root, dirs, files in os.walk(TARGET_DIR):
        level = root.replace(TARGET_DIR, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{subindent}{f}')
else:
    print("Target directory not found. Download or extraction may have failed.")

# Install librosa
install_package("librosa")

# Install soundfile
install_package("soundfile")

TARGET_SR=16000
RESAMPLED_DIR=os.path.join(TARGET_DIR,'resampled_audio')
os.makedirs(RESAMPLED_DIR,exist_ok=True)
print("Resample dir",RESAMPLED_DIR)
print("Target Dir",TARGET_DIR)
input_audio_files=glob.glob(os.path.join(TARGET_DIR,'LibriSpeech','dev-clean','**','*.flac'),recursive=True)
print("Found",len(input_audio_files),"audio files to process")
processed_count=0
failed_count=0
for audio_path in input_audio_files:
  try:
    y,sr_orig=librosa.load(audio_path,sr=None,mono=True)
    if sr_orig!=TARGET_SR:
      y_resampled=librosa.resample(y=y,orig_sr=sr_orig,target_sr=TARGET_SR)
    else:
      y_resampled=y
    relative_path=os.path.relpath(audio_path,os.path.join(TARGET_DIR,'LibriSpeech','dev-clean'))
    output_sub_dir=os.path.join(RESAMPLED_DIR,os.path.dirname(relative_path))
    os.makedirs(output_sub_dir,exist_ok=True)
    output_filename=os.path.basename(audio_path).replace('.flac',f'_{TARGET_SR}hz.wav')
    output_path=os.path.join(RESAMPLED_DIR,output_filename)
    sf.write(output_path,y_resampled,TARGET_SR)
    processed_count+=1
  except Exception as e:
    failed_count+=1
    print(f"Error occur in {audio_path} file:{e}")
print(f"Processed Count :{processed_count}|Failed Count:{failed_count}")

resampled_files = glob.glob(os.path.join(RESAMPLED_DIR, '**', '*.wav'), recursive=True)



NORMALIZED_DIR=os.path.join(TARGET_DIR,'normalized_audio')
os.makedirs(NORMALIZED_DIR,exist_ok=True)
processed_count=0
failed_count=0
for audio_path in resampled_files:
  try:
    y,sr=sf.read(audio_path)
    peak_amplitude=np.max(np.abs(y))
    if peak_amplitude>0:
      target_peak=0.95
      y_normalized=y*(target_peak/peak_amplitude)
    else:
      y_normalized=y
    base_filename=os.path.basename(audio_path).replace(f'_{TARGET_SR}hz.wav','')
    output_filename=f'{base_filename}_normalized.wav'
    output_path=os.path.join(NORMALIZED_DIR,output_filename)
    sf.write(output_path,y_normalized,sr)
    processed_count+=1
  except Exception as e:
    print("Error occur:",e)
    failed_count+=1
print(f"Processed:{processed_count}|Failed:{failed_count}")

normalized_files = glob.glob(os.path.join(NORMALIZED_DIR, '*.wav')) # List files directly in NORMALIZED_DIR


LENGTH_STANDARDIZED_DIR=os.path.join(TARGET_DIR,'length_standaardized_audio')
os.makedirs(LENGTH_STANDARDIZED_DIR,exist_ok=True)

# Clear previously generated length_standardized files
if os.path.exists(LENGTH_STANDARDIZED_DIR):
    shutil.rmtree(LENGTH_STANDARDIZED_DIR)
os.makedirs(LENGTH_STANDARDIZED_DIR,exist_ok=True)

TARGET_AUDIO_DURATION=5
TARGET_SAMPLES=int(TARGET_AUDIO_DURATION*TARGET_SR)

# Get the list of normalized audio files from the previous step
if 'normalized_files' not in locals() or not normalized_files:
    print("No normalized files found. Rescanning...")
    normalized_files = glob.glob(os.path.join(NORMALIZED_DIR, '*.wav'))

processed_count=0
failed_count=0
for audio in normalized_files:
  try:
    y,sr=sf.read(audio) # FIX: Changed audio_path to audio

    if len(y)>TARGET_SAMPLES:
      y_processed=y[:TARGET_SAMPLES]
    elif len(y)<TARGET_SAMPLES:
      padding_needed=TARGET_SAMPLES-len(y)
      y_processed=np.pad(y,(0,padding_needed),'constant')
    else:
      y_processed=y

    base_filename=os.path.basename(audio).replace('_normalized.wav','') # FIX: Changed audio_path to audio
    output_filename=f'{base_filename}_length_standardized.wav'
    output_path=os.path.join(LENGTH_STANDARDIZED_DIR,output_filename)

    sf.write(output_path,y_processed,sr)
    processed_count+=1
  except Exception as e:
    print(f"Error processing {os.path.basename(audio)}: {e}") # FIX: Changed audio_path to audio for error reporting
    failed_count+=1
print(f"Processed:{processed_count}|Failed:{failed_count}")

length_standardized_files = glob.glob(os.path.join(LENGTH_STANDARDIZED_DIR, '*.wav'))



SEGMENTED_DIR = os.path.join(TARGET_DIR, 'segmented_audio')
os.makedirs(SEGMENTED_DIR, exist_ok=True)
SEGMENT_DURATION = 2.5
SEGMENT_SAMPLES = int(SEGMENT_DURATION * TARGET_SR)
OVERLAP_DURATION = 0.5
OVERLAP_SAMPLES = int(OVERLAP_DURATION * TARGET_SR)

processed_count = 0
failed_count = 0

# Get the list of length-standardized audio files from the previous step
if 'length_standardized_files' not in locals() or not length_standardized_files:
    print("No length-standardized files found. Please ensure the length standardization step was executed successfully.")
    # Fallback to re-scanning if length_standardized_files somehow lost its state
    length_standardized_files = glob.glob(os.path.join(LENGTH_STANDARDIZED_DIR, '*.wav'))
    print(f"Found {len(length_standardized_files)} length-standardized files after re-scan.")

# Clear previously generated segments to avoid mixing with corrected ones
if os.path.exists(SEGMENTED_DIR):
    shutil.rmtree(SEGMENTED_DIR)
os.makedirs(SEGMENTED_DIR, exist_ok=True)

for audio_path in length_standardized_files:
    try:
        y, sr = sf.read(audio_path)
        num_samples = len(y)

        # librosa.util.frame returns (frame_length, n_frames). We need to iterate over n_frames.
        frames = librosa.util.frame(y, frame_length=SEGMENT_SAMPLES, hop_length=SEGMENT_SAMPLES - OVERLAP_SAMPLES)

        # Extract a cleaner base filename, stripping the processing suffixes
        original_filename_full = os.path.basename(audio_path)
        # Remove '_length_standardized.wav' and '_16000hz.wav' to get original_base
        clean_filename_base = original_filename_full.replace('_length_standardized.wav', '').replace(f'_{TARGET_SR}hz.wav', '')

        # Iterate over the frames (columns of the frames array)
        for i in range(frames.shape[1]):
            segment = frames[:, i] # Select the i-th frame (column)

            # If the last segment is shorter than SEGMENT_SAMPLES, pad it (shouldn't happen with librosa.util.frame if full length is achieved)
            if len(segment) < SEGMENT_SAMPLES:
                segment = np.pad(segment, (0, SEGMENT_SAMPLES - len(segment)), 'constant')

            output_filename = f"{clean_filename_base}_segment_{i+1}.wav"
            output_path = os.path.join(SEGMENTED_DIR, output_filename)
            sf.write(output_path, segment, sr)
            processed_count += 1

    except Exception as e:
        failed_count += 1
        print(f"Error processing {audio_path}: {e}")

print(f"\nAudio segmentation complete.")
print(f"Processed {processed_count} segments (from original files), failed on {failed_count} original files.")

segmented_files = glob.glob(os.path.join(SEGMENTED_DIR, '*.wav'))

# Define directories for the splits
SPLITS_BASE_DIR = os.path.join(TARGET_DIR, 'audio_splits')
TRAIN_DIR = os.path.join(SPLITS_BASE_DIR, 'train')
VAL_DIR = os.path.join(SPLITS_BASE_DIR, 'validation')
TEST_DIR = os.path.join(SPLITS_BASE_DIR, 'test')

# Create split directories
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

print(f"Train data will be saved to: {TRAIN_DIR}")
print(f"Validation data will be saved to: {VAL_DIR}")
print(f"Test data will be saved to: {TEST_DIR}")

# Get the list of length-standardized audio files.
# We use these as the 'units' to split to avoid data leakage (all segments from one source stay together).
if 'length_standardized_files' not in locals() or not length_standardized_files:
    print("No length-standardized files found. Rescanning...")
    length_standardized_files = glob.glob(os.path.join(LENGTH_STANDARDIZED_DIR, '*.wav'))

if not length_standardized_files:
    raise FileNotFoundError("No length-standardized audio files found to split.")

print(f"Found {len(length_standardized_files)} length-standardized base files for splitting.")

# Step 1: Split length-standardized files into train+val and test sets
train_val_files, test_files = train_test_split(
    length_standardized_files, test_size=0.2, random_state=42, shuffle=True
)

# Step 2: Split train+val into train and validation sets
train_files, val_files = train_test_split(
    train_val_files, test_size=0.25, random_state=42, shuffle=True # 0.25 of 0.8 is 0.2 (20% val)
)


print("\nCopying segmented files to respective split directories...")

train_segment_count = copy_segments_to_split(train_files, TRAIN_DIR, SEGMENTED_DIR)
val_segment_count = copy_segments_to_split(val_files, VAL_DIR, SEGMENTED_DIR)
test_segment_count = copy_segments_to_split(test_files, TEST_DIR, SEGMENTED_DIR)

print("\nDataset split complete.")
print(f"Total segmented files copied: {train_segment_count + val_segment_count + test_segment_count}")
print(f"Train set: {train_segment_count} segments")
print(f"Validation set: {val_segment_count} segments")
print(f"Test set: {test_segment_count} segments")


# 1. Define the root directory of the original LibriSpeech dataset
LIBRISPEECH_ROOT = os.path.join(TARGET_DIR, 'LibriSpeech')

# 2. Initialize an empty dictionary for full transcriptions
full_transcriptions = {}

# 3. Recursively find all .trans.txt files
trans_files = glob.glob(os.path.join(LIBRISPEECH_ROOT, '**', '*.trans.txt'), recursive=True)

print(f"Found {len(trans_files)} transcription files.")

# 4. Iterate through each .trans.txt file and populate full_transcriptions
for trans_file_path in trans_files:
    try:
        with open(trans_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        utterance_id, transcription = parts
                        full_transcriptions[utterance_id] = transcription
                    else:
                        print(f"Skipping malformed line in {trans_file_path}: {line}")
    except Exception as e:
        print(f"Error reading transcription file {trans_file_path}: {e}")

print(f"Loaded {len(full_transcriptions)} unique utterance transcriptions.")

# 5. Initialize dictionary for segmented audio to text mapping
audio_to_text_mapping = {}

# 6. Get a list of all segmented audio files
segmented_audio_files = glob.glob(os.path.join(SEGMENTED_DIR, '*.wav'))

print(f"Found {len(segmented_audio_files)} segmented audio files.")

# 7. Iterate through each segmented audio file and create the mapping
for audio_segment_path in segmented_audio_files:
    try:
        # Extract the base filename (utterance ID) from the segmented file name
        # Example: '1988-24833-0000_segment_1.wav' -> '1988-24833-0000'
        segment_basename = os.path.basename(audio_segment_path)
        # Remove '_segment_X.wav' part to get the original utterance_id
        utterance_id_from_segment = segment_basename.rsplit('_segment_', 1)[0]

        if utterance_id_from_segment in full_transcriptions:
            audio_to_text_mapping[audio_segment_path] = full_transcriptions[utterance_id_from_segment]
        else:
            print(f"Warning: Transcription not found for utterance ID {utterance_id_from_segment} (from {segment_basename}).")
    except Exception as e:
        print(f"Error processing segmented audio file {audio_segment_path}: {e}")


# 1. Collect all unique characters from the full_transcriptions dictionary
all_characters = set()
for transcriptions in full_transcriptions.values():
    all_characters.update(list(transcriptions))

# Define special tokens
special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']

# Create a sorted list of unique characters, excluding special tokens if they accidentally appeared in data
# and then adding them at the beginning for consistent IDs.
vocabulary = sorted(list(all_characters - set(special_tokens))) # Remove any special tokens if they exist in the raw data
vocabulary = special_tokens + vocabulary # Add special tokens at the beginning

# 2. Create character-to-ID mapping (dictionary) and ID-to-character mapping (list)
char_to_id = {char: i for i, char in enumerate(vocabulary)}
id_to_char = {i: char for i, char in enumerate(vocabulary)}

print(f"Vocabulary size: {len(vocabulary)}")
print(f"Sample char_to_id mapping: {dict(list(char_to_id.items())[:10])}")
print(f"Sample id_to_char mapping: {dict(list(id_to_char.items())[:10])}")

# Get IDs for special tokens
PAD_ID = char_to_id['<PAD>']
SOS_ID = char_to_id['<SOS>']
EOS_ID = char_to_id['<EOS>']
UNK_ID = char_to_id['<UNK>']

# 4. Apply the conversion function to all transcriptions in audio_to_text_mapping
#    Store the result in a new dictionary: audio_to_id_sequence_mapping
audio_to_id_sequence_mapping = {}
for audio_path, transcription in audio_to_text_mapping.items():
    id_sequence = text_to_sequence(transcription, char_to_id, SOS_ID, EOS_ID, UNK_ID)
    audio_to_id_sequence_mapping[audio_path] = id_sequence

# 5. Print a few examples for verification
print(f"\nCreated {len(audio_to_id_sequence_mapping)} mappings for ID sequences.")


# Define parameters for Mel spectrogram extraction
N_MELS = 80 # Number of Mel bands to generate
N_FFT = 1024 # Length of the FFT window
HOP_LENGTH = 256 # Number of samples between successive frames

# Directory to store the Mel spectrograms
MEL_SPECTROGRAM_DIR = os.path.join(TARGET_DIR, 'mel_spectrograms')
os.makedirs(MEL_SPECTROGRAM_DIR, exist_ok=True)

print(f"Storing Mel spectrograms in: {MEL_SPECTROGRAM_DIR}")

# Get the list of segmented audio files
# Ensure 'segmented_files' is populated, if not, re-scan
if 'segmented_files' not in locals() or not segmented_files:
    print("No segmented files found. Rescanning...")
    segmented_files = glob.glob(os.path.join(SEGMENTED_DIR, '*.wav'))

if not segmented_files:
    raise FileNotFoundError("No segmented audio files found to process for Mel spectrograms.")

print(f"Found {len(segmented_files)} segmented audio files for Mel spectrogram extraction.")

processed_count = 0
failed_count = 0

# Dictionary to store the mapping of audio_path to mel_spectrogram_path
audio_to_mel_map = {}

for audio_path in segmented_files:
    try:
        # Load the audio waveform
        y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True, res_type='sox_hq')

        # Compute the Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)

        # Convert to decibels
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Create an output filename and path for the spectrogram
        base_filename = os.path.basename(audio_path).replace('.wav', '')
        output_filename = f'{base_filename}_mel.npy'
        output_path = os.path.join(MEL_SPECTROGRAM_DIR, output_filename)

        # Save the Mel spectrogram as a NumPy array
        np.save(output_path, mel_spectrogram_db)
        audio_to_mel_map[audio_path] = output_path
        processed_count += 1
    except Exception as e:
        failed_count += 1
        print(f"Error processing {audio_path}: {e}")

print(f"\nMel spectrogram extraction complete.")
print(f"Processed {processed_count} audio files, failed on {failed_count} files.")
print(f"Stored {len(audio_to_mel_map)} Mel spectrograms in {MEL_SPECTROGRAM_DIR}.")

mel_spectrogram_files = glob.glob(os.path.join(MEL_SPECTROGRAM_DIR, '*.npy'))

class TTSDataset(Dataset):
    def __init__(self, text_id_sequence_map, mel_spectrogram_map, split_dir, vocab_size):
        self.data_pairs = []
        self.vocab_size = vocab_size

        # Filter data pairs relevant to the current split (train, val, or test)
        split_audio_files = [os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith('.wav')]

        for audio_path in split_audio_files:
            if audio_path in text_id_sequence_map and audio_path in mel_spectrogram_map:
                self.data_pairs.append({
                    'audio_path': audio_path,
                    'text_sequence_ids': text_id_sequence_map[audio_path],
                    'mel_spectrogram_path': mel_spectrogram_map[audio_path]
                })
        print(f"Loaded {len(self.data_pairs)} samples for split: {os.path.basename(split_dir)}")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        item = self.data_pairs[idx]
        text_ids = torch.LongTensor(item['text_sequence_ids'])
        mel_spectrogram = torch.FloatTensor(np.load(item['mel_spectrogram_path']))
        return text_ids, mel_spectrogram


# Instantiate TTSDataset for each split, passing SEGMENTED_DIR for correct path lookup
train_dataset = TTSDataset(
    audio_to_id_sequence_mapping,
    audio_to_mel_map,
    TRAIN_DIR,
    SEGMENTED_DIR, # Pass the original segmented directory to resolve paths
    len(vocabulary)
)
val_dataset = TTSDataset(
    audio_to_id_sequence_mapping,
    audio_to_mel_map,
    VAL_DIR,
    SEGMENTED_DIR, # Pass the original segmented directory to resolve paths
    len(vocabulary)
)
test_dataset = TTSDataset(
    audio_to_id_sequence_mapping,
    audio_to_mel_map,
    TEST_DIR,
    SEGMENTED_DIR, # Pass the original segmented directory to resolve paths
    len(vocabulary)
)
class VocoderDataset(Dataset):
    def __init__(self, mel_spectrogram_map, segmented_audio_dir, split_dir):
        self.data_pairs = []

        split_audio_files = [os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith('.wav')]

        for audio_path_in_split_dir in split_audio_files:
            original_segmented_audio_path = os.path.join(segmented_audio_dir, os.path.basename(audio_path_in_split_dir))

            if original_segmented_audio_path in mel_spectrogram_map:
                self.data_pairs.append({
                    'audio_path': original_segmented_audio_path,
                    'mel_spectrogram_path': mel_spectrogram_map[original_segmented_audio_path]
                })
        print(f"Loaded {len(self.data_pairs)} samples for vocoder split: {os.path.basename(split_dir)}")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        item = self.data_pairs[idx]

        mel_spectrogram = torch.FloatTensor(np.load(item['mel_spectrogram_path']))

        y, sr = sf.read(item['audio_path'])
        if sr != TARGET_SR:
            print(f"Warning: Audio {item['audio_path']} has SR {sr} but TARGET_SR is {TARGET_SR}.")
        raw_audio_waveform = torch.FloatTensor(y)

        return mel_spectrogram, raw_audio_waveform

train_vocoder_dataset = VocoderDataset(
    audio_to_mel_map,
    SEGMENTED_DIR,
    TRAIN_DIR
)
val_vocoder_dataset = VocoderDataset(
    audio_to_mel_map,
    SEGMENTED_DIR,
    VAL_DIR
)