import os
import numpy as np
import torch
import torchvision
from torchvision import transforms

def loaddata():
    train_transforms = transforms.Compose([
        transforms.Resize((256//4, 256//4)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((256//4, 256//4)),
        transforms.ToTensor(),
    ])

    root = 'dataset'
    train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(root, 'train'), 
                                                     transform=train_transforms)
    val_dataset = torchvision.datasets.ImageFolder(root=os.path.join(root, 'val'), 
                                                   transform=val_transforms)
    return train_dataset, val_dataset

