import torch.nn as nn

def weights_init(m):
  class_name=m.__class__.__name__
  if class_name.find('Conv')!=-1:
    nn.init.normal_(m.weight.data,0.0,0.02)
  elif class_name.find('BatchNorm')!=-1:
    nn.init.normal_(m.weight.data,1.0,0.02)
    nn.init.constant_(m.bias.data,0)