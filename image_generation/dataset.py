from torchvision.transforms import v2
import torchvision.datasets as datasets
import torch

data_transforms=v2.Compose([
    v2.ToImage(),
    v2.RandomResizedCrop((32,32)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32,scale=True),
    v2.Normalize(mean=[0.5],std=[0.5])
])

train_dataset=datasets.CIFAR10(root="./data",train=True,download=True,transform=data_transforms)
test_dataset=datasets.CIFAR10(root="./data",train=False,download=True,transform=data_transforms)
