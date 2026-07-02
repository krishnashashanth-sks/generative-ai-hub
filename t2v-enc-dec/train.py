from tqdm.auto import tqdm

def train_model(num_epochs,train_dataloader,videotext_generator,optimizer,criterion,device):
    for epoch in tqdm(range(num_epochs)):
        videotext_generator.train() # Set model to training mode
        total_loss = 0

        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            video_sequence = batch['video'].to(device)
            text_sequence = batch['text'].to(device)

            optimizer.zero_grad() # Zero the gradients

            # Forward pass
            generated_video = videotext_generator(video_sequence, text_sequence)

            # Calculate reconstruction loss (ground truth is the input video itself)
            loss = criterion(generated_video, video_sequence)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0: # Print every 10 batches or as desired
                tqdm.write(f"  Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] finished, Average Loss: {avg_loss:.4f}")

    print("Training with increased capacity model complete.")