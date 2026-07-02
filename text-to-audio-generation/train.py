import torch
from tqdm.auto import tqdm
import time
from evaluate import evaluate
import copy

def train_epoch(model,dataloader,criterion,optimizer,device):
  model.train()
  total_loss = 0 # Initialize total_loss
  for batch_idx,(text_batch,text_lengths,mel_batch,mel_lengths) in enumerate(dataloader):
    text_batch=text_batch.to(device)
    text_lengths=text_lengths.to(device)
    mel_batch=mel_batch.to(device)
    mel_lengths=mel_lengths.to(device)

    optimizer.zero_grad()
    current_max_mel_frames=torch.max(mel_lengths).item()
    actual_target_mels=mel_batch[:,:model.max_mel_frames,:]
    predicted_mels=model(text_batch,text_lengths,actual_target_mels)
    loss=0
    for i in range(predicted_mels.size(0)):
      pred_len=predicted_mels.size(1)
      target_len=mel_lengths[i]
      loss+=criterion(predicted_mels[i,:min(pred_len,target_len),:],mel_batch[i,:min(pred_len,target_len),:])
    loss/=predicted_mels.size(0)
    loss.backward()
    optimizer.step()
    total_loss+=loss.item()
  return total_loss/len(dataloader)

def train_model(num_epochs,model,train_loader,val_loader,loss_fn,optimizer,device):
    # Lists to store training and validation losses
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None

    print("Starting training...")

    for epoch in tqdm(range(num_epochs)):
        start_time = time.time()

        # Train for one epoch
        current_train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        train_losses.append(current_train_loss)

        # Evaluate on the validation set
        current_val_loss = evaluate(model, val_loader, loss_fn, device)
        val_losses.append(current_val_loss)

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {current_train_loss:.4f} - "
            f"Validation Loss: {current_val_loss:.4f} - "
            f"Time: {epoch_duration:.2f}s")

        # Save the best model
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  --> Best model saved with Validation Loss: {best_val_loss:.4f}")
    return train_losses,val_losses,best_val_loss,best_model_state