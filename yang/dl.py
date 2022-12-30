import torch, os
from torch import nn
from torch.nn import functional as F
from torch.nn import DataParallel
from torch.cuda import device_count
from sklearn.metrics import roc_auc_score
import numpy as np

from yang.yang import update_json, mkdir

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
                self.matrix[label, int(prob >= 0.5)] += 1
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

class EarlyStopping:
    def __init__(self, net, optimizer, fold='', patience=20, stop_epoch=50, checkpoint_path='.'):
        self.net = net
        self.optimizer = optimizer
        self.fold = fold
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.counter = 0
        self.min_loss = np.Inf
        self.checkpoint_path = os.path.join(checkpoint_path, str(fold))
        mkdir(self.checkpoint_path)
        self.best_epoch = 0

    def __call__(self, logger, epoch):
        if logger.loss < self.min_loss:
            self.min_loss = logger.loss
            self.counter = 0
            self.best_epoch = epoch
            torch.save(self.net.state_dict(), self.best_net_state_dict_name)
            torch.save(self.optimizer.state_dict(), self.best_optimizer_state_dict_name)
            update_json(os.path.join(self.checkpoint_path, 'checkpoints.json'), {epoch: logger.to_dict(['loss', 'acc', 'pos_acc', 'neg_acc', 'f1_score', 'auc'])})
        else:
            self.counter += 1
            if self.counter >= self.patience and epoch >= self.stop_epoch:
                return True
        return False

    @property
    def best_net_state_dict_name(self):
        return os.path.join(self.checkpoint_path, f'net_{self.best_epoch}.pt')

    @property
    def best_optimizer_state_dict_name(self):
        return os.path.join(self.checkpoint_path, f'optimizer_{self.best_epoch}.pt')

def print_net(net):
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

def net_load_state_dict(net, state_dict_name):
    net.load_state_dict(torch.load(state_dict_name))
