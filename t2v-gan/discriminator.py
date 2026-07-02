import torch.nn as nn
import torch

class VideoDiscriminator(nn.Module):
  def __init__(self,in_channels,num_frames,frame_height,frame_width):
    super().__init__()
    self.in_channels=in_channels
    self.num_frames=num_frames
    self.frame_height=frame_height
    self.frame_width=frame_width
    self.main=nn.Sequential(
        nn.Conv3d(in_channels,64,kernel_size=(3,4,4),stride=(2,2,2),padding=(1,1,1),bias=False),
        nn.BatchNorm3d(64),
        nn.LeakyReLU(0.2,inplace=True),
        nn.Conv3d(64,128,kernel_size=(3,4,4),stride=(2,2,2),padding=(1,1,1),bias=False),
        nn.BatchNorm3d(128),
        nn.LeakyReLU(0.2,inplace=True),
        nn.Conv3d(128,256,kernel_size=(3,4,4),stride=(2,2,2),padding=(1,1,1),bias=False),
        nn.BatchNorm3d(256),
        nn.LeakyReLU(0.2,inplace=True),
        nn.Conv3d(256,512,kernel_size=(3,4,4),stride=(2,2,2),padding=(1,1,1),bias=False),
        nn.BatchNorm3d(512),
        nn.LeakyReLU(0.2,inplace=True)
    )
    dummy_input=torch.randn(1,in_channels,num_frames,frame_height,frame_width)
    with torch.no_grad():
      flattened_size=self.main(dummy_input).view(1,-1).shape[1]
    self.output_layer=nn.Linear(flattened_size,1)
  def forward(self,video_frames):
    x=self.main(video_frames)
    x=torch.flatten(x,1)
    output=self.output_layer(x)
    return output