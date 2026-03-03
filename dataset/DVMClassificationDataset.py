from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import torch
from torchvision.io import read_image

class DVMClassificationDataset(Dataset):
    def __init__(self, csv_path, torch_labels, augmentation_rate=0.4):
        self.augmentation_rate = augmentation_rate
        self.csv = torch.load(csv_path, weights_only=False)
        self.labels = torch.load(torch_labels, weights_only=False)
        self.transform = transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.Lambda(lambda x : x.float())
        ])
        img_size=128
        self.transform_augment = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=29, sigma=(0.1, 2.0))],p=0.5),
            transforms.RandomResizedCrop(size=(img_size,img_size), scale=(0.6, 1.0), ratio=(0.75, 1.3333333333333333)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize(size=(img_size,img_size)),
            transforms.Lambda(lambda x : x.float())
        ])

    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, index):
        im = self.csv[index]
        image = read_image(im)
        image = image/255
        label = self.labels[index]
        if random.random() <= self.augmentation_rate:
            image = self.transform_augment(image)
        else:
            image = self.transform(image)
        return {"scan" : image,
                'label' : torch.tensor(label, dtype=torch.long),
                'eid' : im
                }