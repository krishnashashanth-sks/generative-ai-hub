import torch
import torch.nn as nn
class CustomCNNFeatureExtractor(nn.Module):
  def __init__(self,input_channels=3,output_dim=512):
    super(CustomCNNFeatureExtractor,self).__init__()
    self.conv_block1=nn.Sequential(
        nn.Conv2d(input_channels,32,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2)
    )
    self.conv_block2=nn.Sequential(
        nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2)

    )
    self.flatten=nn.Flatten()
    self.fc_layers=nn.Sequential(
      nn.Linear(64*50*50,1024),
      nn.ReLU(),
      nn.Linear(1024,output_dim)
    )
  def forward(self,x):
    return self.fc_layers(self.flatten(self.conv_block2(self.conv_block1(x))))