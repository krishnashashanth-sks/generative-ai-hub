from utils import text_to_sequence
import torch

# --- Define the speech synthesis function ---
def synthesize_speech_end_to_end(text, acoustic_model, vocoder_model, char_to_id, sos_id, eos_id, unk_id, device):
    # 1. Prepare text input for acoustic model
    text_ids = text_to_sequence(text.upper(), char_to_id, sos_id, eos_id, unk_id)
    text_tensor = torch.LongTensor(text_ids).unsqueeze(0).to(device) # Add batch dimension
    
    # Create a dummy mask for inference as FastSpeech2 expects it
    # All True for non-padded parts, and False for padded parts (though we don't pad single inference input)
    phoneme_mask = torch.ones_like(text_tensor, dtype=torch.bool).to(device)

    # 2. Generate Mel spectrogram using acoustic model
    with torch.no_grad():
        predicted_mels, _, _, _ = acoustic_model(text_tensor, phoneme_mask)

    # 3. Synthesize waveform using vocoder
    with torch.no_grad():
        waveform = vocoder_model(predicted_mels)

    # Remove batch dimension and convert to numpy
    waveform_np = waveform.squeeze(0).cpu().numpy()
    return waveform_np
