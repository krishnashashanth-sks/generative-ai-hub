import torch

def train_acoustic_model(num_epochs,fastspeech2_model,train_loader,val_loader,optimizer,mel_criterion,duration_criterion,pitch_criterion,energy_criterion,device):
    # --- 7. Training and Validation Loops ---
    print("\nStarting FastSpeech2 Acoustic Model Training...")

    for epoch in range(num_epochs):
        fastspeech2_model.train() # Set model to training mode
        total_train_mel_loss = 0
        total_train_duration_loss = 0
        total_train_pitch_loss = 0
        total_train_energy_loss = 0

        for batch_idx, (text_inputs, phoneme_mask, duration_target, mel_targets, mel_lengths) in enumerate(train_loader):
            text_inputs, phoneme_mask, duration_target, mel_targets = text_inputs.to(device), phoneme_mask.to(device), duration_target.to(device), mel_targets.to(device)

            optimizer.zero_grad()

            # Forward pass
            predicted_mels, predicted_durations, predicted_pitch, predicted_energy = \
                fastspeech2_model(text_inputs, phoneme_mask, duration_target=duration_target)

            # Pad or trim predicted mels to match target length (due to dynamic length regulation)
            target_mel_frames = mel_targets.shape[2] # (batch, n_mels, frames)
            if predicted_mels.shape[2] != target_mel_frames:
                if predicted_mels.shape[2] > target_mel_frames:
                    predicted_mels = predicted_mels[:, :, :target_mel_frames]
                else:
                    pad_size = target_mel_frames - predicted_mels.shape[2]
                    predicted_mels = torch.nn.functional.pad(predicted_mels, (0, pad_size), 'constant', 0)

            # Loss calculations
            mel_loss = mel_criterion(predicted_mels, mel_targets)
            duration_loss = duration_criterion(predicted_durations, duration_target.float()) # Duration target should be float for MSE

            # Pitch and energy targets are not available from our current data pipeline,
            # so these losses will effectively train the predictors towards 0 or a default, or be skipped.
            # For a true FastSpeech2, these would be derived from ground truth audio.
            # For now, let's include placeholder losses that will be high without proper targets.
            # Alternatively, if we don't have targets, we can choose not to train these predictors directly.
            # For this demonstration, we'll assign dummy targets (e.g., zeros) for pitch/energy loss for now,
            # but ideally, these would come from audio analysis.
            dummy_pitch_target = torch.zeros_like(predicted_pitch).to(device) # Placeholder
            dummy_energy_target = torch.zeros_like(predicted_energy).to(device) # Placeholder

            pitch_loss = pitch_criterion(predicted_pitch, dummy_pitch_target)
            energy_loss = energy_criterion(predicted_energy, dummy_energy_target)

            # Total loss (weights can be adjusted)
            total_loss = mel_loss + duration_loss + pitch_loss + energy_loss

            total_loss.backward()
            optimizer.step()

            total_train_mel_loss += mel_loss.item()
            total_train_duration_loss += duration_loss.item()
            total_train_pitch_loss += pitch_loss.item()
            total_train_energy_loss += energy_loss.item()

            # Clear memory
            del text_inputs, phoneme_mask, duration_target, mel_targets, predicted_mels, predicted_durations, predicted_pitch, predicted_energy, mel_loss, duration_loss, pitch_loss, energy_loss, total_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_train_mel_loss = total_train_mel_loss / len(train_loader)
        avg_train_duration_loss = total_train_duration_loss / len(train_loader)
        avg_train_pitch_loss = total_train_pitch_loss / len(train_loader)
        avg_train_energy_loss = total_train_energy_loss / len(train_loader)

        # Validation Loop
        fastspeech2_model.eval() # Set model to evaluation mode
        total_val_mel_loss = 0
        total_val_duration_loss = 0
        total_val_pitch_loss = 0
        total_val_energy_loss = 0
        with torch.no_grad():
            for batch_idx, (text_inputs, phoneme_mask, duration_target, mel_targets, mel_lengths) in enumerate(val_loader):
                text_inputs, phoneme_mask, duration_target, mel_targets = \
                    text_inputs.to(device), phoneme_mask.to(device), duration_target.to(device), mel_targets.to(device)

                predicted_mels, predicted_durations, predicted_pitch, predicted_energy = \
                    fastspeech2_model(text_inputs, phoneme_mask, duration_target=duration_target)

                target_mel_frames = mel_targets.shape[2]
                if predicted_mels.shape[2] != target_mel_frames:
                    if predicted_mels.shape[2] > target_mel_frames:
                        predicted_mels = predicted_mels[:, :, :target_mel_frames]
                    else:
                        pad_size = target_mel_frames - predicted_mels.shape[2]
                        predicted_mels = torch.nn.functional.pad(predicted_mels, (0, pad_size), 'constant', 0)

                mel_loss = mel_criterion(predicted_mels, mel_targets)
                duration_loss = duration_criterion(predicted_durations, duration_target.float())
                dummy_pitch_target = torch.zeros_like(predicted_pitch).to(device)
                dummy_energy_target = torch.zeros_like(predicted_energy).to(device)
                pitch_loss = pitch_criterion(predicted_pitch, dummy_pitch_target)
                energy_loss = energy_criterion(predicted_energy, dummy_energy_target)

                total_val_mel_loss += mel_loss.item()
                total_val_duration_loss += duration_loss.item()
                total_val_pitch_loss += pitch_loss.item()
                total_val_energy_loss += energy_loss.item()

                # Clear memory
                del text_inputs, phoneme_mask, duration_target, mel_targets, predicted_mels, predicted_durations, predicted_pitch, predicted_energy, mel_loss, duration_loss, pitch_loss, energy_loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        avg_val_mel_loss = total_val_mel_loss / len(val_loader)
        avg_val_duration_loss = total_val_duration_loss / len(val_loader)
        avg_val_pitch_loss = total_val_pitch_loss / len(val_loader)
        avg_val_energy_loss = total_val_energy_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}]\n"
            f"Train Mel Loss: {avg_train_mel_loss:.4f}, Train Dur Loss: {avg_train_duration_loss:.4f}, Train Pitch Loss: {avg_train_pitch_loss:.4f}, Train Energy Loss: {avg_train_energy_loss:.4f}\n"
            f"Val Mel Loss: {avg_val_mel_loss:.4f}, Val Dur Loss: {avg_val_duration_loss:.4f}, Val Pitch Loss: {avg_val_pitch_loss:.4f}, Val Energy Loss: {avg_val_energy_loss:.4f}")

    print("\nFastSpeech2 Acoustic Model Training Complete.")

def train_vocoder_model(num_epochs,vocoder_model,train_vocoder_loader,val_vocoder_loader,optimizer_vocoder,criterion_vocoder,device):
    print("\nStarting HiFiGAN Vocoder Training...")

    for epoch in range(num_epochs):
        vocoder_model.train() # Set model to training mode
        total_train_loss = 0
        for batch_idx, (mel_inputs, waveform_targets) in enumerate(train_vocoder_loader):
            mel_inputs, waveform_targets = mel_inputs.to(device), waveform_targets.to(device)

            optimizer_vocoder.zero_grad()

            predicted_waveforms = vocoder_model(mel_inputs)

            loss = criterion_vocoder(predicted_waveforms, waveform_targets)

            loss.backward()
            optimizer_vocoder.step()
            total_train_loss += loss.item()

            # Explicitly delete variables to free up memory
            del mel_inputs, waveform_targets, predicted_waveforms, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_train_loss = total_train_loss / len(train_vocoder_loader)

        # Validation Loop
        vocoder_model.eval() # Set model to evaluation mode
        total_val_loss = 0
        with torch.no_grad(): # Disable gradient calculation for validation
            for batch_idx, (mel_inputs, waveform_targets) in enumerate(val_vocoder_loader):
                mel_inputs, waveform_targets = mel_inputs.to(device), waveform_targets.to(device)
                predicted_waveforms = vocoder_model(mel_inputs)
                loss = criterion_vocoder(predicted_waveforms, waveform_targets)
                total_val_loss += loss.item()

                # Explicitly delete variables to free up memory
                del mel_inputs, waveform_targets, predicted_waveforms, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        avg_val_loss = total_val_loss / len(val_vocoder_loader)

        print(f"Epoch [{epoch+1}/{num_epochs,vocoder_model,optimizer_vocoder}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    print("\nHiFiGAN Vocoder Training Complete.")