import os
import pandas as pd
import librosa
import numpy as np
import string
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt
from model import build_model
from inference import decode_batch_predictions
from losses import CTCLoss
print("All necessary libraries imported.")

# --- 1. Dataset Download Link---
DATA_DIR = './ljspeech_dataset'
LJSPEECH_URL = 'https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2'
LJSPEECH_ARCHIVE_PATH = os.path.join(DATA_DIR, 'LJSpeech-1.1.tar.bz2')
LJSPEECH_EXTRACTED_DIR = os.path.join(DATA_DIR, 'LJSpeech-1.1')

# --- 2. Load and Process Metadata ---
metadata_path = os.path.join(LJSPEECH_EXTRACTED_DIR, 'metadata.csv')
print(f"Loading metadata from {metadata_path}...")
metadata_df = pd.read_csv(metadata_path, sep='|', header=None)
metadata_df.columns = ['audio_id', 'original_text', 'normalized_text']
print("Metadata loaded.")

# --- 3. Define Audio Processing Parameters ---
TARGET_SAMPLE_RATE = 16000  # Hz
MAX_AUDIO_DURATION = 10     # seconds
MAX_SAMPLES = TARGET_SAMPLE_RATE * MAX_AUDIO_DURATION
N_MFCCS = 13                # Number of MFCCs to extract
N_MELS = 128                # Number of Mel bands to generate
N_FFT = 2048                # Length of the FFT window
HOP_LENGTH = 512            # Number of samples between successive frames

print(f"Audio processing parameters: Sample Rate={TARGET_SAMPLE_RATE}, Max Duration={MAX_AUDIO_DURATION}s")

# Create the full audio file paths
AUDIO_DIR = os.path.join(LJSPEECH_EXTRACTED_DIR, 'wavs')
metadata_df['audio_filepath'] = metadata_df['audio_id'].apply(lambda x: os.path.join(AUDIO_DIR, f'{x}.wav'))

# --- 4. Load and Resample Audio ---
def load_and_resample_audio(audio_path, target_sr=TARGET_SAMPLE_RATE, max_samples=MAX_SAMPLES):
    """Loads an audio file, resamples it to the target sample rate, and trims/pads it to max_samples."""
    try:
        audio, sr = librosa.load(audio_path, sr=None)  # Load with original sample rate
        if sr != target_sr:
            audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)

        if len(audio) > max_samples:
            audio = audio[:max_samples]
        elif len(audio) < max_samples:
            audio = np.pad(audio, (0, max_samples - len(audio)), 'constant')
        return audio
    except Exception as e:
        # print(f"Error processing {audio_path}: {e}") # Uncomment for debugging individual errors
        return None

print("Processing audio files (resampling, padding/trimming)...")
metadata_df['processed_audio'] = metadata_df['audio_filepath'].apply(load_and_resample_audio)

# --- 5. Clean Data: Remove Unprocessed Audio Entries ---
initial_rows = len(metadata_df)
metadata_df = metadata_df.dropna(subset=['processed_audio'])
cleaned_rows = len(metadata_df)
print(f"Removed {initial_rows - cleaned_rows} rows with unprocessed audio. Remaining: {cleaned_rows}")

# --- 6. Preprocess Text Transcripts ---
def normalize_text(text):
    """Normalizes text by lowercasing, removing punctuation, and handling whitespace."""
    if not isinstance(text, str):
        text = str(text)  # Convert non-string types to string
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = ' '.join(text.split())  # Replace multiple spaces with single space and strip leading/trailing
    return text

print("Normalizing text transcripts...")
metadata_df.loc[:, 'processed_text'] = metadata_df['normalized_text'].apply(normalize_text)

# --- 7. Create Vocabulary and Mappings ---
all_characters = set()
for text in metadata_df['processed_text']:
    for char in text:
        all_characters.add(char)

BLANK_TOKEN = ' '  # Using space as blank token
all_characters.add(BLANK_TOKEN)
vocabulary = sorted(list(all_characters))
char_to_int = {char: i for i, char in enumerate(vocabulary)}
int_to_char = {i: char for i, char in enumerate(vocabulary)}

print(f"Vocabulary created. Size: {len(vocabulary)}")

# --- 8. Tokenize Transcripts ---
def tokenize_text(text):
    """Converts a preprocessed transcript string into a sequence of numerical tokens."""
    return [char_to_int[char] for char in text]

print("Tokenizing text transcripts...")
metadata_df.loc[:, 'tokenized_text'] = metadata_df['processed_text'].apply(tokenize_text)

# --- 9. Feature Extraction (MFCCs) ---
def extract_mfccs(audio_data, sr=TARGET_SAMPLE_RATE, n_mfcc=N_MFCCS, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Extracts MFCC features from an audio array."""
    if audio_data is None:
        return None
    try:
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mfccs = librosa.feature.mfcc(S=log_mel_spectrogram, sr=sr, n_mfcc=n_mfcc, dct_type=2)
        return mfccs.T  # Transpose to get (num_frames, n_mfccs)
    except Exception as e:
        # print(f"Error extracting MFCCs: {e}") # Uncomment for debugging individual errors
        return None

print("Extracting MFCC features from audio...")
metadata_df['mfccs'] = metadata_df['processed_audio'].apply(extract_mfccs)

# Clean up failed MFCC extraction
initial_rows_mfcc = len(metadata_df)
metadata_df = metadata_df.dropna(subset=['mfccs'])
cleaned_rows_mfcc = len(metadata_df)
print(f"Removed {initial_rows_mfcc - cleaned_rows_mfcc} rows with failed MFCC extraction. Remaining: {cleaned_rows_mfcc}")

# --- 10. Prepare tf.data.Dataset ---
mfccs_data = np.array(metadata_df['mfccs'].tolist(), dtype=np.float32)
tokenized_text_data = np.array(metadata_df['tokenized_text'].tolist(), dtype=object)

def data_generator():
    for i in range(len(mfccs_data)):
        yield mfccs_data[i].astype(np.float32), np.array(tokenized_text_data[i], dtype=np.int32)

max_text_len = metadata_df['tokenized_text'].apply(len).max()

output_signature = (
    tf.TensorSpec(shape=(None, N_MFCCS), dtype=tf.float32),
    tf.TensorSpec(shape=(None,), dtype=tf.int32)
)

dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=output_signature
)
print("tf.data.Dataset created from processed data.")

# --- 11. Pad and Batch the Dataset ---
BATCH_SIZE = 16
TRAIN_SPLIT_RATIO = 0.8
NUM_MFCC_FRAMES = metadata_df['mfccs'].iloc[0].shape[0]  # Get this after MFCC extraction
dataset_size = len(metadata_df)
train_size = int(TRAIN_SPLIT_RATIO * dataset_size)
val_size = dataset_size - train_size

# Shuffle before splitting for a random split
dataset_shuffled = dataset.shuffle(buffer_size=dataset_size, reshuffle_each_iteration=True)

train_dataset = dataset_shuffled.take(train_size)
val_dataset = dataset_shuffled.skip(train_size)

padded_shapes = (
    tf.TensorShape([NUM_MFCC_FRAMES, N_MFCCS]),  # Fixed shape for MFCCs
    tf.TensorShape([max_text_len])                # Fixed shape for tokenized text
)

padding_values = (
    tf.constant(0.0, dtype=tf.float32),          # Padding value for MFCCs
    tf.constant(char_to_int[BLANK_TOKEN], dtype=tf.int32)  # Padding value for text
)

train_dataset = train_dataset.padded_batch(
    batch_size=BATCH_SIZE,
    padded_shapes=padded_shapes,
    padding_values=padding_values,
    drop_remainder=True
)
val_dataset = val_dataset.padded_batch(
    batch_size=BATCH_SIZE,
    padded_shapes=padded_shapes,
    padding_values=padding_values,
    drop_remainder=True
)

train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

print(f"Dataset split and batched. Train size: {train_size}, Validation size: {val_size}, Batch size: {BATCH_SIZE}")

# --- 12. Define CTCLoss Class ---
ctc_loss_instance = CTCLoss(blank_token_int=char_to_int[BLANK_TOKEN])


# --- 13. Define Model Architecture ---
INPUT_SHAPE = (NUM_MFCC_FRAMES, N_MFCCS)

model=build_model(INPUT_SHAPE,vocabulary)

model.summary()

# --- 14. Compile and Train Model ---
model.compile(optimizer=Adam(learning_rate=0.001), loss=ctc_loss_instance)

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint_filepath = './best_asr_model_weights.h5'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

print("Starting model training...")
history = model.fit(
    train_dataset,
    epochs=20,
    validation_data=val_dataset,
    callbacks=[early_stopping_callback, model_checkpoint_callback]
)

print("Model training complete.")

# --- Optional: Evaluate the model (after training) ---
print("\nEvaluating the model on the validation dataset...")
val_loss = model.evaluate(val_dataset, verbose=1)
print(f"Model Validation Loss: {val_loss:.4f}")

# --- Optional: Plot training history ---
if history is not None:
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Progression')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
# 15.Inference
print("\n--- Demonstrating Inference ---")
# Take a few samples from the validation dataset for inference demonstration
num_inference_samples = 5
sample_mfcc_batches = []
sample_true_texts = []

for mfcc_batch, text_batch in val_dataset.take(1):
    sample_mfcc_batches.append(mfcc_batch.numpy())
    for i in range(min(num_inference_samples, BATCH_SIZE)):
        true_text_tokens = text_batch[i].numpy()
        true_text = "".join([int_to_char[t] for t in true_text_tokens if t != char_to_int[BLANK_TOKEN]])
        sample_true_texts.append(true_text)
    break # Take only the first batch

if sample_mfcc_batches:
    sample_mfcc_batch = sample_mfcc_batches[0][:num_inference_samples]

    # Predict logits
    predictions = model.predict(sample_mfcc_batch)

    # Apply softmax to get probabilities for decoding
    predictions_softmax = tf.nn.softmax(predictions)

    # Decode predictions
    decoded_texts = decode_batch_predictions(
        [predictions_softmax],
        int_to_char,
        char_to_int[BLANK_TOKEN]
    )

    print(f"Displaying {len(decoded_texts)} inference results:")
    for i in range(len(decoded_texts)):
        print(f"--- Sample {i+1} ---")
        print(f"True Text: {sample_true_texts[i]}")
        print(f"Predicted: {decoded_texts[i]}")

        # Add debugging information: check blank token probability
        blank_token_prob = predictions_softmax[i, :, char_to_int[BLANK_TOKEN]].numpy().mean()
        print(f"  Average blank token probability for sample {i+1}: {blank_token_prob:.4f}")

else:
    print("No validation samples available for inference demonstration.")
