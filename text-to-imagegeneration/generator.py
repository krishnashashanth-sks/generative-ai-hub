import torch.nn as nn
import torch

class Generator(nn.Module):
  def __init__(self,nz=100,text_embedding_dim=768):
    super(Generator,self).__init__()
    self.nz=nz
    self.text_embedding_dim=text_embedding_dim
    self.main=nn.Sequential(
        nn.ConvTranspose2d(nz*text_embedding_dim,512,4,2,1,bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.ConvTranspose2d(512,256,4,2,1,bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.ConvTranspose2d(256,128,4,2,1,bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.ConvTranspose2d(128,3,4,2,1,bias=False),
        nn.Tanh()
    )
  def forward(self,noise,text_embedding):
    # Reshape text_embedding for concatenation with noise
    # noise: (batch_size, nz, 1, 1)
    # text_embedding: (batch_size, text_embedding_dim)
    # Reshape text_embedding to (batch_size, text_embedding_dim, 1, 1)
    text_embedding_reshaped=text_embedding.unsqueeze(-1).unsqueeze(-1)
    # Concatenate noise and text_embedding_reshaped along the channel dimension
    input_tensor=torch.cat([noise,text_embedding_reshaped],1)
    return self.main(input_tensor)