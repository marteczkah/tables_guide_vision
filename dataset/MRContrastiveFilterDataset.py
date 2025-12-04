import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import random
import torch
from utils.attributes import CAD_ATTRIBUTES_PRE, ATTRIBUTES
from utils.utils import normalize_scan
import h5py

class MRContrastiveFilterDataset(Dataset):
    def __init__(self, h5_path, csv_path, attribute="", augmentation_rate=-1):
        self.csv = pd.read_csv(csv_path)
        target_eids = self.csv['eid'].to_numpy()

        self.info = h5py.File(h5_path, 'r')
        self.metadata_columns = list(self.info['metadata'].attrs['columns'])
        metadata_ds = self.info['metadata']

        eid_ind = self.metadata_columns.index('eid')
        cad_ind = self.metadata_columns.index('no_cad')

        # Read all EIDs and CAD values from metadata
        h5_eids = metadata_ds[:, eid_ind]
        h5_cads = metadata_ds[:, cad_ind]

        # Filter indices where HDF5 eid is in the CSV eid
        mask = np.isin(h5_eids, target_eids)
        self.filtered_indices = np.where(mask)[0]

        # Store the filtered EIDs and CADs
        self.eid = h5_eids[self.filtered_indices]
        self.cad = h5_cads[self.filtered_indices]
        self.metadata = metadata_ds[self.filtered_indices]

        self.augmentation_rate = augmentation_rate
        self.transform_augment = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(128, scale=(0.6, 1.0)),
                transforms.RandomRotation(45),
            ]
        )

        self.attribute = attribute

        if len(attribute) > 0:
            attribute_ind = self.metadata_columns.index(attribute)
            h5_attribute = metadata_ds[:, attribute_ind]
            self.attributes = h5_attribute[self.filtered_indices]


    def __len__(self):
        return len(self.eid)

    def __getitem__(self, index):
        h5_index = self.filtered_indices[index]

        # Read the image and normalize
        scan_org = self.info["images"][h5_index]
        scan_org = normalize_scan(scan_org)
        scan_org = torch.from_numpy(np.array(scan_org))

        if random.random() <= self.augmentation_rate:
            scan_org = self.transform_augment(scan_org)

        # Build metadata row
        metadata_row = {col: self.metadata[index, i] for i, col in enumerate(self.metadata_columns)}
        attributes = torch.tensor([metadata_row[i] for i in ATTRIBUTES])
        attributes = torch.nan_to_num(attributes, nan=0.0)
        
        multilabel = [metadata_row[i] for i in CAD_ATTRIBUTES_PRE]
        multilabel = torch.tensor(np.nan_to_num(multilabel, nan=0), dtype=torch.float32)
        # CAD label
        cad = self.cad[index]
        if cad == 1:
            is_cad = torch.tensor(0, dtype=torch.float32)
        else:
            is_cad = torch.tensor(1, dtype=torch.float32)
        
        if len(self.attribute) > 0:
            attr = self.attributes[index]
            attr = torch.tensor(np.nan_to_num(attr, nan=0), dtype=torch.float32)
        else:
            attr = torch.tensor(0, dtype=torch.float32)
        return {
            "scan": scan_org.float(),
            "eid": self.eid[index],
            "attributes": attributes,
            "is_cad": is_cad,
            'label':multilabel,
            'attribute': attr
        }