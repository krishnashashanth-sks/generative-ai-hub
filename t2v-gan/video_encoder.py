import torch.nn as nn
import torch

class VideoEncoder(nn.Module):
  def __init__(self,embedding_dim,frame_height,frame_width):
    super().__init__()
    self.cnn_extractor=nn.Sequential(
        nn.Conv2d(3,64,kernel_size=4,stride=2,padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(True),
        nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(True),
        nn.Conv2d(256,512,kernel_size=4,stride=2,padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(True),
        nn.Conv2d(512,512,kernel_size=4,stride=2,padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(True),
        nn.AdaptiveAvgPool2d((1,1))
    )
    dummy_input=torch.randn(1,3,frame_height,frame_width)
    with torch.no_grad():
      features_output_size=self.cnn_extractor(dummy_input).view(1,-1).shape[1]
    self.video_projection=nn.Linear(features_output_size,embedding_dim)
  def forward(self,video_frames):
    batch_size,num_frames,C,H,W=video_frames.shape
    reshaped_frames=video_frames.view(batch_size*num_frames,C,H,W)
    cnn_features=self.cnn_extractor(reshaped_frames)
    cnn_features=cnn_features.view(batch_size,num_frames,-1)
    pooled_video_features=torch.mean(cnn_features,dim=1)
    video_embeddings=self.video_projection(pooled_video_features)
    return video_embeddings