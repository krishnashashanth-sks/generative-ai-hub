import torch.nn as nn

class Generator(nn.Module):
  def __init__(self,nz=100):
    super(Generator,self).__init__()
    self.main=nn.Sequential(
        nn.ConvTranspose2d(nz,512,4,2,1,bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(True),
        nn.ConvTranspose2d(512,256,4,2,1,bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(True),
        nn.ConvTranspose2d(256,128,4,2,1,bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.ConvTranspose2d(128,3,4,2,1,bias=False),
        nn.Tanh()
    )
  def forward(self,x):
    return self.main(x)