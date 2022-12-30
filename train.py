import torch, h5py, os
from torch import nn
from torch.nn import DataParallel
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from yang.yang import mkdir, dict2json
from yang.dl import WithNone, Logger, EarlyStopping, get_all_devices, net2devices, set_seed, net_load_state_dict
from yang.nets import CLAM

seed = 1053532442

# devices = get_all_devices() 
devices = ['cuda:0']

# net_state_dict_name = './checkpoints/checkpoints1/1/net_50.pt'
# optimizer_state_dict_name = './checkpoints/checkpoints1/1/optimizer_50.pt'

CLAM_path = '/home/yangxuan/CLAM/'

is_ihc = True
if is_ihc:
    exp_name = 'IHC/exp3'
    label_path = '/home/yangxuan/dataset/IHC/label.csv'
    feature_path = '/home/yangxuan/CLAM/features/IHC/h5_files/'
    args = {
        'max_epoch': 50,
        'net': {
            'type': 'CLAM', 
        },
        'optimizer': {
            'type': 'AdamW', # ['Adam', 'AdamW']
            'lr': 1e-4, 
            'weight_decay': 1e-4
        },
        # 'lr_scheduler': {
            # 'type': 'OneCycleLR', 
            # 'lr_times': 10
        # },
        'lr_scheduler': {
            'type': 'None'
        },
    }
else:
    exp_name = 'HE/exp2'
    label_path = '/home/yangxuan/dataset/HE/label.csv'
    feature_path = '/home/yangxuan/CLAM/features/HE/h5_files/'
    args = {
        'max_epoch': 100,
        'net': {
            'type': 'CLAM', 
        },
        'optimizer': {
            'type': 'AdamW', # ['Adam', 'AdamW']
            'lr': 1e-4, 
            'weight_decay': 1e-4
        },
        'lr_scheduler': {
            'type': 'None'
        },
    }

log_path = os.path.join(CLAM_path, 'log', exp_name)
checkpoint_path = os.path.join(CLAM_path, 'checkpoint', exp_name)
final_res_path = os.path.join(CLAM_path, 'final_res', exp_name)
split_path = os.path.join(CLAM_path, 'split', exp_name)
arg_path = os.path.join(CLAM_path, 'arg', exp_name)

mkdir(log_path)
mkdir(checkpoint_path)
mkdir(final_res_path)
mkdir(split_path)
mkdir(arg_path)

final_res_name = os.path.join(final_res_path, 'final_res.json')

class WSI_Dataset(Dataset):
    def __init__(self, file_names, labels):
        self.file_names = file_names
        self.labels = labels

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name, label = self.file_names[idx], self.labels[idx]
        with h5py.File(os.path.join(feature_path, file_name + '.h5'), 'r') as f:
            return torch.from_numpy(f['features'][:]), label

    def to_csv(self, csv_name, probs=None):
        d = {'file_name': self.file_names, 'label': self.labels}
        if probs:
            print(self.labels)
            d.update({'prob': [(prob if label else 1 - prob) for prob, label in zip(probs, self.labels)]})
        pd.DataFrame(d, index=list(range(1, len(self.file_names) + 1))).to_csv(csv_name)

def collate_MIL(batch):
	features = torch.cat([item[0] for item in batch], dim=0)
	labels = torch.LongTensor([item[1] for item in batch])
	return [features, labels]

def do_epoch(is_train: bool, loader, net, loss_fn, optimizer=None, lr_scheduler=None):
    net.train() if is_train else net.eval()

    with WithNone() if is_train else torch.no_grad():
        res = Logger()
        for i, (x, y) in enumerate(loader):
            x, y = x.to(devices[0]), y.to(devices[0])

            logits = net(x)

            loss = loss_fn(logits.unsqueeze(dim=0), y)
            probs = F.softmax(logits, dim=0)
            res.add(loss.item(), y.item(), probs[1].item())

            if is_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if lr_scheduler:
                    lr_scheduler.step()

            if (i + 1) % 10 == 0:
                print(f'batch: {i + 1}/{len(loader)}')

        return res

class FinalLogger():
    def __init__(self, attrs):
        self.data = {}
        for attr in attrs:
            self.data[attr] = []
        self.best_epochs = []

    def update(self, logger, best_epoch):
        for attr in self.data.keys():
            self.data[attr].append(eval(f'logger.{attr}.item()'))
        self.best_epochs.append(best_epoch)

    def get_res_dict(self):
        res = {'best_epochs': self.best_epochs}
        for attr in self.data.keys():
            res[attr] = {}
            res[attr]['data'] = self.data[attr]
            res[attr]['min'] = min(self.data[attr])
            res[attr]['mean'] = sum(self.data[attr]) / len(self.data[attr])
            res[attr]['max'] = max(self.data[attr])
        print(res)

        return res

def read_split(split_name):
    csv = pd.read_csv(split_name)
    return list(csv['file_name']), list(csv['label'])

def train():
    df = pd.read_csv(label_path, header=None)
    file_names, labels = np.array(df[0], dtype=str), np.array(df[1], dtype=np.uint8)
    skf = StratifiedKFold(5)

    final_logger = FinalLogger(['loss', 'acc', 'pos_acc', 'neg_acc', 'f1_score', 'auc'])

    # save_args
    dict2json(os.path.join(arg_path, 'arg.json'), args)

    for fold, (train_i, val_i) in enumerate(skf.split(file_names, labels)):
        # seed
        set_seed(seed)

        # Dataset DataLoader
        train_dataset = WSI_Dataset(file_names[train_i], labels[train_i])
        train_loader = DataLoader(train_dataset, batch_size=1, sampler=RandomSampler(train_dataset), collate_fn=collate_MIL)
        val_dataset = WSI_Dataset(file_names[val_i], labels[val_i])
        val_loader = DataLoader(val_dataset, batch_size=1, sampler=SequentialSampler(val_dataset), collate_fn=collate_MIL)

        # writer
        writer_path = os.path.join(log_path, str(fold + 1))
        mkdir(writer_path)
        writer = SummaryWriter(writer_path)

        # net
        if args['net']['type'] == 'CLAM':
            net = net2devices(CLAM(), devices)
        # if net_state_dict_name:
            # net_load_state_dict(net, net_state_dict_name)
        # if optimizer_state_dict_name:
            # net_load_state_dict(optimizer, optimizer_state_dict_name)

        # loss
        loss_fn = nn.CrossEntropyLoss()

        # optimizer
        if args['optimizer']['type'] == 'Adam':
            optimizer = Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args['optimizer']['lr'], weight_decay=args['optimizer']['weight_decay'])
        elif args['optimizer']['type'] == 'AdamW':
            optimizer = AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=args['optimizer']['lr'], betas=(0.9, 0.999), eps=1e-8, weight_decay=args['optimizer']['weight_decay'], amsgrad=False)

        # lr_scheduler
        if args['lr_scheduler']['type'] == 'OneCycleLR':
            lr_scheduler = OneCycleLR(optimizer=optimizer, max_lr=args['optimizer']['lr'] * args['lr_scheduler']['lr_times'], epochs=args['max_epoch'], steps_per_epoch=len(train_loader), pct_start=0.3, div_factor=20, final_div_factor=1000)
        else:
            lr_scheduler = None

        early_stopping = EarlyStopping(net=net, optimizer=optimizer, fold=fold + 1, patience=20, stop_epoch=args['max_epoch'], checkpoint_path=checkpoint_path)

        for epoch in range(args['max_epoch']):
            train_res = do_epoch(True, train_loader, net, loss_fn, optimizer)
            val_res = do_epoch(False, val_loader, net, loss_fn)

            print(f'epoch:{epoch + 1}')
            print(f'train: {train_res}')
            print(f'val: {val_res}')

            print(train_res.matrix)
            print(val_res.matrix)

            if writer:
                train_res.to_writer(writer, 'train/', ['loss', 'acc', 'pos_acc', 'neg_acc', 'f1_score', 'auc'], epoch + 1)
                val_res.to_writer(writer, 'val/', ['loss', 'acc', 'pos_acc', 'neg_acc', 'f1_score', 'auc'], epoch + 1)

            if early_stopping(val_res, epoch + 1):
                print('early_stopping at {epoch + 1} epoch')
                break

        net_load_state_dict(net, early_stopping.best_net_state_dict_name)
        final_res = do_epoch(False, val_loader, net, loss_fn)
        final_logger.update(final_res, early_stopping.best_epoch)

        train_dataset.to_csv(os.path.join(split_path, f'train_{fold + 1}.csv'))
        val_dataset.to_csv(os.path.join(split_path, f'val_{fold + 1}.csv'), final_res.probs)

        if writer:
            val_res.to_writer(writer, 'final/', ['loss', 'acc', 'pos_acc', 'neg_acc', 'f1_score', 'auc'], fold + 1)

    dict2json(final_res_name, final_logger.get_res_dict())

if __name__ == '__main__':
    train()
