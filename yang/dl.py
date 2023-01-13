import torch, os, h5py
from torch import nn
from torch.nn import functional as F
from torch.nn import DataParallel
from torch.cuda import device_count
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

from yang.yang import mkdir

def get_all_devices():
    return [f'cuda:{i}' for i in range(device_count())] if device_count() else ['cpu']

def net2devices(net, devices):
    return net.to(devices[0]) if len(devices) == 1 else DataParallel(net.to(devices[0]), devices)

def get_net_one_device(net):
    if isinstance(net, DataParallel):
        net = net.module
    for para in net.parameters():
        return para.device

def get_net_all_devices(net):
    devices = set([])
    if isinstance(net, DataParallel):
        net = net.module
    for para in net.parameters():
        devices.add(para.device)
    return devices

class WithNone():
    def __enter__(self):
        pass
    def __exit__(self, err_type, err_val, err_pos):
        pass

class Logger():
    def __init__(self):
        self.loss_sum = 0
        self.labels = []
        self.probs = []
        # matrix[i][j] --> matrix[label][pred]
        # TN, FP
        # FN, TP
        self.matrix = torch.zeros((2, 2), dtype=torch.int)

    def add(self, loss, labels, probs):
        self.loss_sum += loss
        if torch.is_tensor(labels):
            self.labels.extend(labels.tolist())
            self.probs.extend(probs.tolist())
            for label, prob in zip(labels, probs):
                self.matrix[int(label), int(prob >= 0.5)] += 1
        else:
            self.labels.append(labels)
            self.probs.append(probs)
            self.matrix[labels, int(probs >= 0.5)] += 1

    def __str__(self):
        return f'loss: {self.loss}, acc: {self.acc}, pos_acc: {self.pos_acc}, neg_acc: {self.neg_acc}, f1_score: {self.f1_score}, auc: {self.auc}'

    @property
    def loss(self):
        return self.loss_sum / self.matrix.sum()

    @property
    def acc(self):
        return (self.matrix[0, 0] + self.matrix[1, 1]) / self.matrix.sum()

    @property
    def pos_acc(self):
        return self.matrix[1, 1] / self.matrix[1, :].sum()

    @property
    def neg_acc(self):
        return self.matrix[0, 0] / self.matrix[0, :].sum()

    @property
    def precision(self):
        return self.matrix[1, 1] / self.matrix[:, 1].sum()
    
    @property
    def recall(self):
        return self.matrix[1, 1] / self.matrix[1, :].sum()

    @property
    def f1_score(self):
        return 2 * self.precision * self.recall / (self.precision + self.recall)

    @property
    def auc(self):
        return roc_auc_score(self.labels, self.probs)

    def to_writer(self, writer, prefix, attrs, epoch):
        for attr in attrs:
            writer.add_scalar(prefix + attr, eval(f'self.{attr}'), epoch)

    def to_dict(self, attrs):
        d = {}
        for attr in attrs:
            d[attr] = eval(f'self.{attr}').item()
        return d

class EarlyStop:
    def __init__(self, patience=20, min_stop_epoch=50, max_stop_epoch=200):
        self.patience = patience
        self.min_stop_epoch = min_stop_epoch
        self.max_stop_epoch = max_stop_epoch
        self.counter = 0
        self.min_loss = np.Inf

        self.stop_epoch = 0
        self.best_epoch = 0
        self.early_stop = False

    def __call__(self, loss, epoch):
        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1

        if self.counter > self.patience and epoch >= self.min_stop_epoch or epoch == self.max_stop_epoch:
            self.stop_epoch = epoch
            self.early_stop = True

class FinalLogger():
    def __init__(self, attrs):
        self.data = {}
        for attr in attrs:
            self.data[attr] = []
        self.stop_epochs = []
        self.best_epochs = []

    def update(self, logger, stop_epoch, best_epoch):
        for attr in self.data.keys():
            self.data[attr].append(eval(f'logger.{attr}.item()'))
        self.stop_epochs.append(stop_epoch)
        self.best_epochs.append(best_epoch)

    def get_res_dict(self):
        res = {
            'stop_epochs': self.stop_epochs,
            'best_epochs': self.best_epochs
        }
        for attr in self.data.keys():
            res[attr] = {}
            res[attr]['data'] = self.data[attr]
            res[attr]['min'] = min(self.data[attr])
            res[attr]['mean'] = sum(self.data[attr]) / len(self.data[attr])
            res[attr]['max'] = max(self.data[attr])

        return res

def print_net_parameters(net):
    print(net)

    num_params, num_train_params = 0, 0
    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_train_params += n

    print(f'Total num of parameters: {num_params}')
    print(f'Total num of trainable parameters: {num_train_params}')

def set_seed(seed):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def read_split(split_name):
    csv = pd.read_csv(split_name)
    return list(csv['file_name']), list(csv['label'])

def read_h5(feature_name, key):
    with h5py.File(feature_name, 'r') as f:
        return f[key][:]

def do_epoch(is_train: bool, loader, net, loss_fn, optimizer=None, lr_scheduler=None):
    net.train() if is_train else net.eval()
    device = get_net_one_device(net)

    with WithNone() if is_train else torch.no_grad():
        res = Logger()
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

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

# label.csv --> (file_names, labels)
def read_label_file(label_file_name):
    df = pd.read_csv(label_file_name, index_col=0)
    return np.array(df['file_name'], dtype=str), np.array(df['label'], dtype=np.uint8)

def save_label_file(label_file_name, file_names, labels):
    df = pd.DataFrame({'file_name': file_names, 'label': labels}, index=range(1, len(file_names) + 1))
    df.to_csv(label_file_name)

def print_net_activations(net, x):
    from yang.net import NetExtractor
    extractor = NetExtractor(net, regist_grad=False)
    extractor(x)
    for name, val in extractor.activations.items():
        print(name, val.shape)

def net_load_state_dict(net, state_dict_name):
    net.load_state_dict(torch.load(state_dict_name))
