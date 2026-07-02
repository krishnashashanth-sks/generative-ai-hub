import torch

def evaluate(model,dataloader,criterion,device):
  model.eval()
  total_loss=0
  with torch.no_grad():
    for batch_idx,(text_batch,text_lengths,mel_batch,mel_lengths) in enumerate(dataloader):
      text_batch=text_batch.to(device)
      text_lengths=text_lengths.to(device)
      mel_batch=mel_batch.to(device)
      mel_lengths=mel_lengths.to(device)
      current_max_mel_frames=torch.max(mel_lengths).item()
      # Note: actual_max_mel_frames is defined but not used. It's also redundant if model.max_mel_frames is handled appropriately.
      # actual_max_mel_frames=mel_batch[:,:model.max_mel_frames,:]
      predicted_mels=model(text_batch,text_lengths,None)
      loss=0
      for i in range(predicted_mels.size(0)):
        pred_len=predicted_mels.size(1)
        target_len=mel_lengths[i]
        # Fix: Correctly index mel_batch for each item in the batch
        loss+=criterion(predicted_mels[i,:min(pred_len,target_len),:],mel_batch[i,:min(pred_len,target_len),:])
      loss/=predicted_mels.size(0)
      total_loss+=loss.item()
  return total_loss/len(dataloader)