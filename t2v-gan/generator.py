import torch.nn as nn
import torch

class VideoGenerator(nn.Module):
  def __init__(self,text_embedding_dim,latent_dim,output_channels,num_frames,frame_height,frame_width):
    super().__init__()
    self.text_embedding_dim=text_embedding_dim
    self.latent_dim=latent_dim
    self.output_channels=output_channels
    self.num_frames=num_frames
    self.frame_height=frame_height
    self.frame_width=frame_width
    self.initial_depth=2
    self.initial_height=4
    self.initial_width=4
    self.initial_channels=512
    self.fc=nn.Linear(text_embedding_dim+latent_dim,self.initial_channels*self.initial_depth*self.initial_height*self.initial_width)
    self.main=nn.Sequential(
        nn.ConvTranspose3d(self.initial_channels,256,kernel_size=(3,4,4),stride=(2,2,2),padding=(1,1,1),output_padding=(1,0,0)),
        nn.BatchNorm3d(256),
        nn.ReLU(True),
        nn.ConvTranspose3d(256,128,kernel_size=(3,4,4),stride=(2,2,2),padding=(1,1,1),output_padding=(1,0,0)),
        nn.BatchNorm3d(128),
        nn.ReLU(True),
        nn.ConvTranspose3d(128,64,kernel_size=(3,4,4),stride=(2,2,2),padding=(1,1,1),output_padding=(1,0,0)),
        nn.BatchNorm3d(64),
        nn.ReLU(True),
        nn.ConvTranspose3d(64,output_channels,kernel_size=(1,4,4),stride=(1,4,4),padding=(0,0,0),output_padding=(0,0,0)),
        nn.Tanh()
    )
  def forward(self,text_embeddings,noise_vector):
    combined_input=torch.cat([text_embeddings,noise_vector],dim=1)
    x=self.fc(combined_input)
    x=x.view(-1,self.initial_channels,self.initial_depth,self.initial_height,self.initial_width)
    generated_video=self.main(x)
    return generated_video