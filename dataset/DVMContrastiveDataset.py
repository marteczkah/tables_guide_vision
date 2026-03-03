from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import csv
from torchvision.io import read_image

class DVMContrastiveDataset(Dataset):
    def __init__(self, tab_data_path, data_paths, augmentation_rate=0.4):
        self.augmentation_rate = augmentation_rate
        self.root = '/home/iml/marta.hasny'
        self.tabular = self.read_and_parse_csv(tab_data_path)
        self.paths = torch.load(data_paths, weights_only=False)
        self.label_type = [1, 1, 1, 1, 1, 1, 1, 1, 1, 13, 3, 12, 286]
        self.is_continuous = np.where(np.array(self.label_type) == 1)
        self.is_categorical = np.where(np.array(self.label_type) != 1)
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
        return len(self.paths)
    
    def read_and_parse_csv(self, path):
        """
        Does what it says on the box.
        """
        with open(path,'r') as f:
            reader = csv.reader(f)
            data = []
            for r in reader:
                r2 = [float(r1) for r1 in r]
                data.append(r2)
        return data
    
    def __getitem__(self, index):
        image = read_image(self.paths[index])
        image = image/255
        tabular = np.array(self.tabular[index])
        continuous = tabular[self.is_continuous]
        categorical = self.encode_categorical(tabular[self.is_categorical])
        if random.random() <= self.augmentation_rate:
            image = self.transform_augment(image)
        else:
            image = self.transform(image)
            
        return {"scan" : image, # called scan to match the UKBB training
                'continuous' :torch.tensor(continuous, dtype=torch.float32),
                'categorical' : torch.tensor(categorical, dtype=torch.float32),
                }
    
    def encode_categorical(self, categorical):
        num_cats = np.array(self.label_type)[self.is_categorical]
        encoded = []
        for i, val in enumerate(categorical):
            if num_cats[i] > 1:
                one_hot = [-1] * num_cats[i]
                one_hot[int(val)]=1
                encoded.extend(one_hot)
            else:
                encoded.append(1 if val == 1 else -1)
        return encoded