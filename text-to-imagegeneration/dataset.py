from torch.utils.data import Dataset
import torch

class Text2ImageDataset(Dataset):
  def __init__(self,dataset,transform_image_fn):
    self.dataset=dataset['train']
    self.transform_image_fn=transform_image_fn
  def __len__(self):
    return len(self.dataset)
  def __getitem__(self,idx):
    item=self.dataset[idx] # Corrected from self.datasets
    image_path=item['image_path']
    input_ids=torch.tensor(item['input_ids'],dtype=torch.long)
    attention_mask=torch.tensor(item['attention_mask'],dtype=torch.long)
    transformed_image=self.transform_image_fn(image_path)
    if transformed_image is None:
      # If image loading/transformation fails, try the next item
      return self.__getitem__((idx+1)%len(self))
    return {
        'transformed_image':transformed_image,
        'input_ids':input_ids,
        'attention_mask':attention_mask
    }