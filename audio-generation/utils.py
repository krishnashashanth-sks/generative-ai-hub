import sys
import subprocess
import glob
import os
import torch
import shutil
from dataset import TARGET_SR,PAD_ID,SEGMENT_SAMPLES

def install_package(package):
    try:
        __import__(package)
        print(f"{package} is already installed.")
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"{package} installed successfully.")

# Helper function to get all segments corresponding to a base file
def get_segments_for_base_file(base_file_path, segmented_dir):
    # Extract the base filename without the path and all processing suffixes
    original_filename_full = os.path.basename(base_file_path)
    # This logic should match the clean_filename_base in the segmentation step (SyrW31QoMnL9)
    clean_filename_base = original_filename_full.replace('_length_standardized.wav', '').replace(f'_{TARGET_SR}hz.wav', '').replace('_normalized.wav', '')

    # Segments are named like: clean_filename_base_segment_1.wav
    segment_pattern = f"{clean_filename_base}_segment_*.wav"
    return glob.glob(os.path.join(segmented_dir, segment_pattern))

# Function to copy files to target directory
def copy_segments_to_split(base_file_list, target_dir, segmented_dir):
    count = 0
    for base_file in base_file_list:
        segments = get_segments_for_base_file(base_file, segmented_dir)
        for segment_path in segments:
            shutil.copy(segment_path, os.path.join(target_dir, os.path.basename(segment_path)))
            count += 1
    return count

def tts_collate_fn(batch):
    # Sort the batch by text sequence length in descending order for packing
    batch = sorted(batch, key=lambda x: x[0].size(0), reverse=True)

    text_sequences, mel_spectrograms = zip(*batch)

    # Pad text sequences
    text_lengths = torch.LongTensor([len(seq) for seq in text_sequences])
    padded_text_sequences = torch.nn.utils.rnn.pad_sequence(
        text_sequences, batch_first=True, padding_value=PAD_ID # PAD_ID is from previous step
    )

    # Mel spectrograms are already padded to SEGMENT_SAMPLES in earlier step, so we just stack them.
    # However, since they might have different frame lengths due to slight variations after librosa.util.frame and TARGET_AUDIO_DURATION, we need to handle this.
    # For simplicity, we'll pad Mel spectrograms to the max length in the batch as well.
    mel_lengths = torch.LongTensor([mel.size(1) for mel in mel_spectrograms]) # Assuming mel.shape is (n_mels, frames)
    # Pad mel spectrograms (first dimension is n_mels, second is frames)
    # We need to pad the second dimension (frames)
    max_mel_len = max(mel_lengths)
    padded_mel_spectrograms = []
    for mel in mel_spectrograms:
        pad_size = max_mel_len - mel.size(1)
        if pad_size > 0:
            # Pad along the second dimension (frames)
            padded_mel = torch.nn.functional.pad(mel, (0, pad_size), 'constant', 0)
        else:
            padded_mel = mel
        padded_mel_spectrograms.append(padded_mel)
    padded_mel_spectrograms = torch.stack(padded_mel_spectrograms)

    return padded_text_sequences, text_lengths, padded_mel_spectrograms, mel_lengths

def vocoder_collate_fn(batch):
    mel_spectrograms, raw_audio_waveforms = zip(*batch)

    # Pad mel spectrograms to the max length in the batch
    mel_lengths = torch.LongTensor([mel.size(1) for mel in mel_spectrograms])
    max_mel_len = max(mel_lengths)
    padded_mel_spectrograms = []
    for mel in mel_spectrograms:
        pad_size = max_mel_len - mel.size(1)
        if pad_size > 0:
            padded_mel = torch.nn.functional.pad(mel, (0, pad_size), 'constant', 0)
        else:
            padded_mel = mel
        padded_mel_spectrograms.append(padded_mel)
    stacked_mels = torch.stack(padded_mel_spectrograms)

    # Pad raw audio waveforms to the max length in the batch (or a fixed SEGMENT_SAMPLES)
    # Assuming all raw_audio_waveforms are SEGMENT_SAMPLES long from preprocessing
    # If they are not, we need to pad/truncate them here.
    # For this simplified setup, we expect them to be consistent.
    # max_waveform_len = max([len(w) for w in raw_audio_waveforms])
    # However, SEGMENT_SAMPLES is a fixed value so we should just pad to that.
    padded_raw_audio_waveforms = []
    for waveform in raw_audio_waveforms:
        if len(waveform) > SEGMENT_SAMPLES:
            padded_waveform = waveform[:SEGMENT_SAMPLES]
        elif len(waveform) < SEGMENT_SAMPLES:
            pad_size = SEGMENT_SAMPLES - len(waveform)
            padded_waveform = torch.nn.functional.pad(waveform, (0, pad_size), 'constant', 0)
        else:
            padded_waveform = waveform
        padded_raw_audio_waveforms.append(padded_waveform)
    stacked_waveforms = torch.stack(padded_raw_audio_waveforms)

    return stacked_mels, stacked_waveforms

def text_to_sequence(text, char_to_id, sos_token_id, eos_token_id, unk_token_id):
    sequence = [sos_token_id] # Start with SOS token
    for char in text:
        sequence.append(char_to_id.get(char, unk_token_id)) # Use UNK for unknown characters
    sequence.append(eos_token_id) # End with EOS token
    return sequence