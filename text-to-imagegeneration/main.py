import os
# Use this to download dataset http://images.cocodataset.org/annotations/annotations_trainval2017.zip
#!pip install transformers

from transformers import AutoTokenizer
from datasets import load_dataset
import json
from dataset import Text2ImageDataset
from torch.utils.data import DataLoader
from transform import load_and_transform_image
import torch
import matplotlib.pyplot as plt
from train import train
from generator import Generator
from discriminator import Discriminator
import torch.optim as optim
from utils import weights_init
import torch.nn as nn

tokenizer=AutoTokenizer.from_pretrained('openai/clip-vit-base-patch32')

def tokenizer_captions(batch):
  return tokenizer(batch['caption'],padding='max_length',truncation=True,max_length=77,return_tensors='pt')

# Load the COCO annotations JSON file
annotations_file_path = 'annotations/captions_train2017.json'

with open(annotations_file_path, 'r') as f:
    coco_annotations = json.load(f)

# Create a simple dataset structure for captions
# Each image has multiple captions, so we'll flatten this structure for tokenization
caption_data = []
for annotation in coco_annotations['annotations']:
    caption_data.append({
        'image_id': annotation['image_id'],
        'caption': annotation['caption']
    })

# Convert to a Hugging Face Dataset
dataset = load_dataset('json', data_files={'train': annotations_file_path}, field='annotations')

tokenizer_datasets=dataset.map(tokenizer_captions,batched=True)

image_id_to_path={}
base_dir='train2017'

for image_info in coco_annotations['images']:
  image_id=image_info['id']
  file_name=image_info['file_name']
  full_path=os.path.join(base_dir,file_name)
  image_id_to_path[image_id]=full_path
print(f"Created mapping for {len(image_id_to_path)}")

if image_id_to_path:
  sample_id=next(iter(image_id_to_path))
  print(f"Sample mapping :ID{sample_id}->Path :{image_id_to_path[sample_id]}")

def add_image_path(example):
  image_id=example['image_id']
  example['image_path']=image_id_to_path.get(image_id)
  return example

dataset_with_paths=tokenizer_datasets.map(add_image_path,batched=False)

custom_dataset=Text2ImageDataset(dataset_with_paths,load_and_transform_image)
batch_size=32
num_workers=os.cpu_count()
data_loader=DataLoader(
    custom_dataset,shuffle=True,
    num_workers=num_workers,
    batch_size=batch_size,
    pin_memory=True
)

clip_model = tokenizer
nz=100
device='cuda' if torch.cuda.is_available() else 'cpu'

fixed_noise=torch.randn(batch_size,nz,1,1,device=device)

# Define fixed text embedding for visualization
sample_caption_for_viz = "A dog jumping over a fence" # Example fixed caption
tokenized_fixed_caption = tokenizer(sample_caption_for_viz, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
fixed_input_ids_viz = tokenized_fixed_caption['input_ids'].to(device)
fixed_attention_mask_viz = tokenized_fixed_caption['attention_mask'].to(device)

with torch.no_grad():
    clip_model.eval() # Ensure CLIP model is in evaluation mode
    single_fixed_text_embedding = clip_model.get_text_features(
        input_ids=fixed_input_ids_viz,
        attention_mask=fixed_attention_mask_viz
    )
    # Expand the single embedding to match the batch size of fixed_noise
    fixed_text_embedding = single_fixed_text_embedding.repeat(fixed_noise.size(0), 1)

text_embedding_dim=512
netG=Generator(nz,text_embedding_dim)
netD=Discriminator(3,64,text_embedding_dim)
criterion=nn.BCELoss()

netG.apply(weights_init)
netD.apply(weights_init)
optimizerD=optim.Adam(netD.parameters(),lr=0.0001,betas=(0.5,0.999))
optimizerG=optim.Adam(netG.parameters(),lr=0.002,betas=(0.5,0.999))

epochs=10

losses_G,losses_D,D_x_list,D_G_z_list,img_list=train(epochs,data_loader,netG,netD,clip_model,criterion,optimizerG,optimizerD,nz,fixed_noise,fixed_text_embedding,device)


# Plot the training metrics
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(losses_G,label="G")
plt.plot(losses_D,label="D")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.title("D(x) and D(G(z)) During Training")
plt.plot(D_x_list,label="D(x)")
plt.plot(D_G_z_list,label="D(G(z))")
plt.xlabel("Iterations")
plt.ylabel("Probability")
plt.legend()
plt.show()

# Visualize generated images from img_list
if img_list:
    fig = plt.figure(figsize=(12,12))
    plt.axis("off")
    # Display the last few generated image grids or a selection
    num_grids_to_show = min(len(img_list), 5) # Show at most 5 grids
    for k in range(num_grids_to_show):
        plt.subplot(1, num_grids_to_show, k+1)
        plt.imshow(img_list[-(num_grids_to_show-k)].permute(1,2,0)) # Display in HWC format
        plt.axis("off")
    plt.suptitle("Generated Images over Training Progress (Fixed Noise)", fontsize=16)
    plt.show()
else:
    print("No images were saved during training for visualization.")
