import torch, h5py, os, openslide
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from yang.dl import read_h5


class WSI_Dataset(Dataset):
    def __init__(self, file_names, labels, feature_path):
        self.file_names = file_names
        self.labels = labels
        self.feature_path = feature_path

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name, label = self.file_names[idx], self.labels[idx]
        return torch.from_numpy(read_h5(os.path.join(self.feature_path, file_name + '.h5'), 'features')), label

    def to_csv(self, csv_name, probs=None):
        d = {'file_name': self.file_names, 'label': self.labels}
        if probs:
            d.update({'prob': [(prob if label else 1 - prob) for prob, label in zip(probs, self.labels)]})
        pd.DataFrame(d, index=list(range(1, len(self.file_names) + 1))).to_csv(csv_name)

    @staticmethod
    def collate_fn(batch):
        features = torch.cat([item[0] for item in batch], dim=0)
        labels = torch.LongTensor([item[1] for item in batch])
        return [features, labels]

def WSI_Loader(dataset, is_train: bool):
    return DataLoader(dataset, batch_size=1, sampler=RandomSampler(dataset) if is_train else SequentialSampler(dataset), collate_fn=dataset.collate_fn)

class PatchDataset(Dataset):
    def __init__(self, wsi_name, patch_name, transform):
        self.wsi = openslide.open_slide(wsi_name)
        with h5py.File(patch_name, 'r') as f:
            dset = f['coords']
            self.coords = dset[:]
            self.patch_level = dset.attrs['patch_level']
            self.patch_size = dset.attrs['patch_size']

        self.transform = transform

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        img = self.wsi.read_region(self.coords[idx], self.patch_level, (self.patch_size, ) * 2).convert('RGB') 
        img = self.transform(img).unsqueeze(0)
        return img, self.coords[idx]

    @staticmethod
    def collate_fn(batch):
        img = torch.cat([item[0] for item in batch], dim=0)
        coords = np.vstack([item[1] for item in batch])
        return [img, coords]

def PatchLoader(dataset, batch_size, num_workers=4):
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, collate_fn=dataset.collate_fn)
