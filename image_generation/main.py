import os
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import train_dataset,test_dataset
from generator import Generator 
from discriminator import Discriminator
import torch
from utils import weights_init
import torch.nn as nn
from train import train

nz=100
netG=Generator(nz)
netD=Discriminator()

train_dataloader=DataLoader(train_dataset,batch_size=64,shuffle=True)
test_dataloader=DataLoader(test_dataset,batch_size=64,shuffle=False)
netG.apply(weights_init)
netD.apply(weights_init)

device='cuda'if torch.cuda.is_available() else 'cpu'
netD.to(device)
netG.to(device)
optimizerD=torch.optim.Adam(netD.parameters(),lr=0.0002,betas=(0.5,0.999))
optimizerG=torch.optim.Adam(netG.parameters(),lr=0.002,betas=(0.5,0.999))
loss_fn=nn.BCELoss()

epochs=100

losses_G,losses_D,img_list=train(epochs,nz,train_dataloader,netD,netG,optimizerD,optimizerG,loss_fn,device)


# Create a directory to save generated images
output_dir = 'generated_images'
os.makedirs(output_dir, exist_ok=True)
print(f"Created directory: {output_dir}")


# Plot the training metrics
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(losses_G,label="G")
plt.plot(losses_D,label="D")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Visualize a few generated images at the end of training
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(vutils.make_grid(img_list[idx], padding=5, normalize=True).permute(1,2,0))] for idx in range(len(img_list))]
plt.show()