from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import torch
from torchvision.io import read_image

class DVMImageDataset(Dataset):
    def __init__(self, csv_path, ids_path):
        self.csv = torch.load(csv_path, weights_only=False)
        self.eid = torch.load(ids_path, weights_only=False)
        self.transform = transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.Lambda(lambda x : x.float())
        ])

    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, index):
        im = self.csv[index]
        image = read_image(im)
        image = image/255
        image = self.transform(image)
        eid = self.eid[index]
        return {"scan" : image,
                'eid' : eid
                }