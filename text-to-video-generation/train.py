import torch.nn.functional as F

def train(num_epochs,model,data_loader,optimizer,criterion,device):
    for epoch in range(num_epochs):
        model.train()
        total_loss=0
        for i,(video_batch,caption_batch) in enumerate(data_loader):
            video_batch=video_batch.to(device)
            caption_batch=caption_batch.to(device)

            optimizer.zero_grad()

            generated_videos=model(caption_batch)

            # Dynamically resize generated_videos to match video_batch's dimensions
            # video_batch.shape[2:] gives (frames, height, width)
            generated_videos = F.interpolate(generated_videos, size=video_batch.shape[2:], mode='trilinear', align_corners=False)

            loss=criterion(generated_videos,video_batch)
            loss.backward()
            optimizer.step()

            total_loss+=loss.item()

        avg_loss=total_loss/len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        print("Training loop finished.")