import torch
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def synthesize_mel_spectrogram(model, text, char_to_idx, device, max_text_len=None):
    """
    Performs inference to synthesize a Mel-spectrogram from input text.

    Args:
        model (nn.Module): The trained TextToAudioModel.
        text (str): The raw input text to synthesize.
        char_to_idx (dict): Character to index mapping for tokenization.
        device (torch.device): The device (CPU or CUDA) to run inference on.
        max_text_len (int, optional): Maximum text sequence length for padding. If None, uses actual length.

    Returns:
        np.ndarray: The generated Mel-spectrogram as a NumPy array.
    """
    model.eval()  # Set model to evaluation mode
    # Temporarily disable teacher forcing for inference
    original_teacher_forcing_ratio = model.teacher_forcing_ratio
    model.teacher_forcing_ratio = 0.0

    # 1. Preprocess text
    preprocessed_text = preprocess_text(text)
    print(f"Preprocessed text for inference: '{preprocessed_text}'")

    # 2. Tokenize text
    tokenized_text = [char_to_idx[char] for char in preprocessed_text if char in char_to_idx]
    if not tokenized_text:
        print("Warning: No valid tokens found for the input text.")
        return None
    
    # Convert to tensor and add batch dimension
    text_tensor = torch.tensor(tokenized_text, dtype=torch.long, device=device).unsqueeze(0)
    input_lengths = torch.tensor([len(tokenized_text)], dtype=torch.long, device=device)

    # 3. Perform forward pass
    with torch.no_grad():
        # During inference, target_mel_spectrograms is None
        predicted_mel_frames = model(text_tensor, input_lengths, target_mel_spectrograms=None)

    # Restore original teacher forcing ratio
    model.teacher_forcing_ratio = original_teacher_forcing_ratio
    model.train() # Set model back to train mode if it was originally in train mode

    # Convert the output Mel-spectrogram to NumPy array and remove batch dimension
    generated_mel = predicted_mel_frames.squeeze(0).cpu().numpy()
    return generated_mel