import librosa
import numpy as np
import re
import string

def preprocess_audio(audio_path, target_sr):
    """
    Loads an audio file, resamples it to the target sample rate, and normalizes its amplitude.

    Args:
        audio_path (str): Path to the audio file.
        target_sr (int): Target sample rate.

    Returns:
        np.ndarray: Preprocessed audio waveform.
    """
    # Load audio with original sample rate to avoid resampling twice if not needed
    # sr=None loads the audio at its native sampling rate
    y, sr = librosa.load(audio_path, sr=None)

    # Resample to target_sr if the original sample rate is different
    if sr != target_sr:
        y = librosa.resample(y=y, orig_sr=sr, target_sr=target_sr)

    # Normalize amplitude to [-1, 1]
    # librosa.util.normalize is a common way, or manually by dividing by max(abs(y))
    y = librosa.util.normalize(y)

    return y

def preprocess_text(text):
    """
    Normalizes text by lowercasing, removing punctuation, and normalizing whitespace.

    Args:
        text (str): The input text transcript.

    Returns:
        str: The normalized text.
    """
    text = text.lower() # Lowercase all characters
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text) # Remove punctuation
    text = re.sub(r'\s+', ' ', text) # Normalize whitespace
    text = text.strip() # Remove leading/trailing whitespace

    # Further normalization (optional, depending on model requirements)
    # e.g., expanding numbers like "123" to "one hundred twenty-three"
    # For this basic example, we'll keep it simple.

    return text

def audio_to_mel_spectrogram(audio_waveform, sr, n_mels=80, n_fft=2048, hop_length=512):
    """
    Converts an audio waveform into a Mel-spectrogram.

    Args:
        audio_waveform (np.ndarray): The preprocessed audio waveform.
        sr (int): The sampling rate of the audio waveform.
        n_mels (int): Number of Mel bands to generate.
        n_fft (int): Length of the FFT window.
        hop_length (int): Number of samples between successive frames.

    Returns:
        np.ndarray: The Mel-spectrogram in decibels.
    """
    # Compute the Mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_waveform, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)

    # Convert to decibel scale
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return mel_spectrogram_db