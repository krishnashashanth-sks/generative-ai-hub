import torch.nn as nn
import torch

class Discriminator(nn.Module):
  def __init__(self,nc=3,ndf=64,text_embedding_dim=768):
    super(Discriminator,self).__init__()
    self.text_embedding_dim=text_embedding_dim
    # First block: Image processing before text embedding injection
    # Input: (nc, 32, 32)
    self.initial_layers=nn.Sequential(
        nn.Conv2d(nc,ndf,4,2,1,bias=False),
        nn.BatchNorm2d(ndf),
        nn.LeakyReLU(0.2,inplace=True),
        nn.Conv2d(ndf,ndf*2,4,2,1,bias=False),
        nn.BatchNorm2d(ndf*2),
        nn.LeakyReLU(0.2,inplace=True),
        nn.Conv2d(ndf*2,ndf*4,4,2,1,bias=False),
        nn.BatchNorm2d(ndf*4),
        nn.LeakyReLU(0.2,inplace=True),
    )
    # Final block: Combines image features with text embedding for classification
    # Input channels will be (ndf * 4 + text_embedding_dim)
    # The output from initial_layers is (batch_size, ndf*4, 4, 4)
    self.final_layers=nn.Sequential(
        nn.Conv2d(ndf*4+self.text_embedding_dim,1,4,1,0,bias=False),
        nn.Sigmoid()
    )
  def forward(self,image,text_embedding):
    image_features=self.initial_layers(image)
    # Reshape text_embedding to match spatial dimensions of image_features
    # text_embedding: (batch_size, text_embedding_dim)
    # We want it to be (batch_size, text_embedding_dim, 4, 4) for concatenation
    # Use .expand to broadcast the text embedding across the spatial dimensions
    text_embedding_reshaped=text_embedding.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,image_features.size(2),image_features.size(3))
    combined_input=torch.cat([image_features,text_embedding_reshaped],1)
    return self.final_layers(combined_input).view(-1) # Flatten to (batch_size) for BCELoss