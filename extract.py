import openslide, h5py, os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.cuda import device_count
from yang.yang import get_file_names_by_suffix
from torchvision.models import resnet50, ResNet50_Weights
from torch.nn import DataParallel
import torch
import numpy as np
from models.resnet_custom import resnet50_baseline

devices = [f'cuda:{i}' for i in range(device_count())] if device_count() else ['cpu']

class WSI_Bag(Dataset):
    def __init__(self, wsi_name, patch_name):
        self.wsi = openslide.open_slide(wsi_name)
        with h5py.File(patch_name, 'r') as f:
            dset = f['coords']
            self.coords = dset[:]
            self.patch_level = dset.attrs['patch_level']
            self.patch_size = dset.attrs['patch_size']

        self.transform = Compose([ToTensor(), Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        img = self.wsi.read_region(self.coords[idx], self.patch_level, (self.patch_size, ) * 2).convert('RGB') 
        img = self.transform(img).unsqueeze(0)
        return img, self.coords[idx]

def collate_fn(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    coords = np.vstack([item[1] for item in batch])
    return [img, coords]

def save_h5(h5_name, data_dict, attr_dict=None):
    with h5py.File(h5_name, 'a') as f:
        for key, val in data_dict.items():
            if key not in f:
                dset = f.create_dataset(key, shape=val.shape, maxshape=(None, ) + val.shape[1:], chunks=(1, ) + val.shape[1:], data=val)
                if attr_dict and key in attr_dict.keys():
                    dset.attrs.update(attr_dict[key])
            else:
                dset = f[key]
                dset.resize(len(dset) + val.shape[0], axis=0)
                dset[-val.shape[0]:] = val

def extract_features(wsi_name, patch_name, out_name, net):
    dataset = WSI_Bag(wsi_name, patch_name)
    loader = DataLoader(dataset, batch_size=1024, num_workers=4, pin_memory=True, collate_fn=collate_fn)

    for i, (patches, coords) in enumerate(loader):
        with torch.no_grad():
            patches = patches.to(devices[0], non_blocking=True)
            features = net(patches).cpu().numpy()

            save_h5(out_name, {'features': features, 'coords': coords})

            print(f'extract_features:{i + 1}/{len(loader)}')

if __name__ == '__main__':
    # wsi_path = '/home/yangxuan/dataset/IHC/'
    # patch_path = '/home/yangxuan/code/python/CLAM/results/IHC/patches/'
    # out_path = '/home/yangxuan/data/features/IHC/h5_files'
    wsi_path = '/home/yangxuan/dataset/HE/'
    patch_path = '/home/yangxuan/CLAM/patches/HE_30/patches/'
    out_path = '/home/yangxuan/CLAM/features/HE/'
    file_names = get_file_names_by_suffix(patch_path, '.h5')

    # net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    net = resnet50_baseline(pretrained=True)
    net = net.to(devices[0]) if len(devices) == 1 else DataParallel(net.to(devices[0]), devices)
    net.eval()

    err = 0
    for i, file_name in enumerate(file_names):
        wsi_name = os.path.join(wsi_path, file_name + '.svs')
        patch_name = os.path.join(patch_path, file_name + '.h5')
        out_name = os.path.join(out_path, file_name + '.h5')

        # if os.path.exists(out_name):
            # print('already exists')
            # print(f'{i + 1}/{len(file_names)}')
            # continue

        try:
            extract_features(wsi_name, patch_name, out_name, net)
        except Exception as e:
            print(e)
            err += 1

        print(f'{i + 1}/{len(file_names)}')

    print(f'err:{err}')
