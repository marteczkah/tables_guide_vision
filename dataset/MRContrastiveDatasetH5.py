import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import random
import torch
from utils.attributes import CAT_LABELS, CATEGORICAL_ATTRIBUTES, NUMERICAL_ATTRIBUTES, NUMERICAL_MAPPING, CAD_ATTRIBUTES_PRE, ATTRIBUTES
import h5py
from utils.utils import normalize_scan

class MRContrastiveDatasetH5(Dataset):
    def __init__(self, h5_path, augment_org = False, attribute="", augmentation_rate=-1, root='/lustre/groups/shared/ukbb-87065/dataset/cardiac_mri_nifti'):
        self.root = root
        self.info = h5py.File(h5_path, 'r')
        self.metadata_columns = self.info['metadata'].attrs['columns']
        self.metadata = np.array(self.info['metadata'])
        eid_ind = list(self.metadata_columns).index('eid')
        cad_ind = list(self.metadata_columns).index('no_cad')
        self.eid = self.info['metadata'][:, eid_ind]
        self.cad = self.info['metadata'][:, cad_ind]
        self.augment_org = augment_org
        self.transform_augment = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(128, scale=(0.6, 1.0)),
                transforms.RandomRotation(45),
             ]
        )
        # self.CAD = info[CAD_ATTRIBUTES_POST].to_numpy()
        self.augmentation_rate = augmentation_rate
        self.attribute = attribute

        if len(attribute) > 0:
            attribute_ind = list(self.metadata_columns).index(attribute)
            self.attributes = self.metadata[:, attribute_ind]

    def __len__(self):
        return len(self.eid)
    
    def __getitem__(self, index):
        scan_org = self.info["images"][index]
        scan_org = normalize_scan(scan_org)
        if self.augment_org:
            if random.random() <= self.augmentation_rate:
                scan_org = torch.from_numpy(np.array(scan_org))
                scan_org = self.transform_augment(scan_org)
            else:
                scan_org = torch.from_numpy(np.array(scan_org))
        else:
                scan_org = torch.from_numpy(np.array(scan_org))
        metadata_row = {col: self.metadata[index, i] for i, col in enumerate(ATTRIBUTES)}
        categorical = torch.tensor(self.encode_categorical(metadata_row), dtype=torch.float32).unsqueeze(0)
        continuous = torch.tensor([metadata_row[i] for i in NUMERICAL_ATTRIBUTES], dtype=torch.float32).unsqueeze(0)
        continuous = torch.from_numpy(np.nan_to_num(continuous, nan=0))
        continuous = torch.nn.functional.normalize(continuous, p=2, dim=1)
        cad = self.cad[index]
        multilabel = [metadata_row[i] for i in CAD_ATTRIBUTES_PRE]
        multilabel = torch.tensor(np.nan_to_num(multilabel, nan=0), dtype=torch.float32)
        if cad == 1:
            has_cad = torch.tensor(0, dtype=torch.float32)
        else:
            has_cad = torch.tensor(1, dtype=torch.float32)
        
        if len(self.attribute) > 0:
            attr = self.attributes[index]
            attr = torch.tensor(np.nan_to_num(attr, nan=0), dtype=torch.float32)
        else:
            attr = torch.tensor(0, dtype=torch.float32)
        return {"scan": scan_org, 
                "continuous": continuous,
                'categorical': categorical,
                "eid": self.eid[index],
                'has_cad':has_cad,
                'label':multilabel,
                'attribute':attr}
    
    def encode_categorical(self, attributes):
        num_cats = np.array(CAT_LABELS)
        encoded = []
        i = 0
        for key, value in attributes.items():
            if key in CATEGORICAL_ATTRIBUTES:
                if pd.isna(value):
                    value = 0
                if num_cats[i] > 2:
                    one_hot = [-1] * num_cats[i]
                    # ind = NUMERICAL_MAPPING[key][int(value)] # what did I do with smoking that it doesn't need mapping
                    ind = int(value)
                    one_hot[ind] = 1
                    encoded.extend(one_hot)
                else:
                    if int(value) == 1:
                        encoded.append(1)
                    else:
                        encoded.append(-1)
                i += 1
        return encoded
