import openslide, h5py, os, timm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.cuda import device_count
from torchvision.models import resnet50, ResNet50_Weights
from torch.nn import DataParallel
import torch
import numpy as np

from models.resnet_custom import resnet50_baseline

from yang.yang import get_file_names_by_suffix, mkdir, rmdir
from yang.dl import net2devices, get_all_devices
from yang.dataset import PatchDataset, PatchLoader
from yang.net import CTransPath

# devices = get_all_devices()
devices = ['cuda:0']

task = 'HE'

if task == 'IHC': 
    exp_name = 'ResNet50'
    args = {
            'net': {
                'type': 'ResNet50', # ['ResNet50', 'CTransPath']
            }, 
            'batch_size': 1024, 
            'transform': 'resize', # ['resize', 'None']
    }
elif task == 'HE':
    exp_name = 'no_xml/ResNet50'
    args = {
            'net': {
                'type': 'ResNet50', # ['ResNet50', 'CTransPath']
            }, 
            'batch_size': 1024,
            'transform': 'None', # ['resize', 'None']
    }

wsi_path = os.path.join('/home/yangxuan/dataset/', task)
patch_path = os.path.join('/home/yangxuan/CLAM/patches/', task, task)
out_path = os.path.join('/home/yangxuan/CLAM/features/', task, exp_name)
mkdir(out_path)

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

def init_patch_loader(wsi_name):
    transform = [Resize((224, 224))] if args['transform'] == 'resize' else []
    transform.extend([ToTensor(), Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    dataset = PatchDataset(wsi_name, patch_name, Compose(transform))
    return PatchLoader(dataset, args['batch_size'], 4)

def extract_features(wsi_name, patch_name, out_name, net):
    loader = init_patch_loader(wsi_name)
    for i, (patches, coords) in enumerate(loader):
        with torch.no_grad():
            patches = patches.to(devices[0], non_blocking=True)
            features = net(patches).cpu().numpy()

            save_h5(out_name, {'features': features, 'coords': coords})

            print(f'extract_features:{i + 1}/{len(loader)}')

def init_net():
    if args['net']['type'] == 'ResNet50':
        net = net2devices(resnet50_baseline(pretrained=True), devices)
    elif args['net']['type'] == 'CTransPath':
        net = net2devices(CTransPath(), devices)
    net.eval()

    return net

if __name__ == '__main__':
    file_names = get_file_names_by_suffix(patch_path, '.h5')

    net = init_net()

    err = 0
    for i, file_name in enumerate(file_names):
        wsi_name = os.path.join(wsi_path, file_name + '.svs')
        patch_name = os.path.join(patch_path, file_name + '.h5')
        out_name = os.path.join(out_path, file_name + '.h5')

        try:
            extract_features(wsi_name, patch_name, out_name, net)
        except Exception as e:
            print(e)
            err += 1

        print(f'{i + 1}/{len(file_names)}')

    print(f'err:{err}')
