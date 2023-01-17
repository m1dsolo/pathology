import torch, os, itertools
from torch import nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import OneCycleLR, MultiStepLR

from yang.yang import mkdirs, rmdirs, dict2json
from yang.dl import net2devices, net_load_state_dict
from yang.net import CLAM_0, CLAM_1, CLAM_2, DTFD_T1, DTFD_T2, Attention, DimReduction, Classifier

seed = 1053532442

# devices = get_all_devices() 
devices = ['cuda:0']

CLAM_path = '/home/yangxuan/CLAM/'

task = 'IHC'

if task == 'IHC':
    exp_name = 'tmp'
    args = {
        # 'net': {
            # 'type': 'DTFD',
            # 'pbag_num': 5, # pseudo_bag_num
            # 'distill': 'AFS', # ['MaxS', 'MaxMinS', 'AFS']
            # 'grad_clipping': 5,
        # },
        'net': {
            'type': 'CLAM_2', 
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
        'early_stop': {
            'patience': 20,
            'min_stop_epoch': 100,
            'max_stop_epoch': 200
        },
        'features': {
            'type': 'ResNet50', # ['ResNet50', 'CTransPath']
        },
    }
elif task == 'HE':
    exp_name = 'ResNet50_CLAM_2'
    args = {
        # 'net': {
            # 'type': 'DTFD',
            # 'pbag_num': 5, # pseudo_bag_num
            # 'distill': 'AFS', # ['MaxS', 'MaxMinS', 'AFS']
            # 'grad_clipping': 5,
        # },
        'net': {
            'type': 'CLAM_2', 
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
        'early_stop': {
            'patience': 20,
            'min_stop_epoch': 100,
            'max_stop_epoch': 200
        },
        'features': {
            'type': 'ResNet50', # ['ResNet50', 'CTransPath']
            'use_xml': False,
        }
    }
elif task == 'camelyon16':
    exp_name = 'DTFD_pbag5_AFS'
    args = {
        'net': {
            'type': 'DTFD',
            'pbag_num': 5, # pseudo_bag_num
            'distill': 'AFS', # ['MaxS', 'MaxMinS', 'AFS']
            'grad_clipping': 5,
        },
        # 'net': {
            # 'type': 'CLAM_2', 
        # },
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
            'type': 'MultiStepLR',
            'milestones': [100],
            'gamma': 0.2
        },
        # 'lr_scheduler': {
            # 'type': 'None'
        # },
        'early_stop': {
            'patience': 20,
            'min_stop_epoch': 160,
            'max_stop_epoch': 200, 
        },
    }

def init_feature_path():
    if task == 'IHC':
        return os.path.join('/home/yangxuan/CLAM/features/IHC', args['features']['type'])
    elif task == 'HE':
        return os.path.join('/home/yangxuan/CLAM/features/HE', 'xml' if args['features']['use_xml'] else 'no_xml', args['features']['type'])
    elif task == 'camelyon16':
        return '/home/sdb/yudan/PROCESSED_DATA/CAMELYON16/FEATURES_DIRECTORY_LEVEL0_resnet50/h5_files'

feature_path = init_feature_path()
train_label_file_name = os.path.join('/home/yangxuan/dataset', task, 'label_train.csv')
test_label_file_name = os.path.join('/home/yangxuan/dataset', task, 'label_test.csv')

exp_name = os.path.join(task, exp_name)
log_path = os.path.join(CLAM_path, 'log', exp_name)
checkpoint_path = os.path.join(CLAM_path, 'checkpoint', exp_name)
res_path = os.path.join(CLAM_path, 'res', exp_name)
split_path = os.path.join(CLAM_path, 'split', exp_name)

rmdirs([log_path, checkpoint_path, res_path, split_path])
mkdirs([log_path, checkpoint_path, res_path, split_path])

# save_args
dict2json(os.path.join(res_path, 'arg.json'), args)

def load_checkpoint(fold, nets, net_names):
    for net, net_name in zip(nets, net_names):
        net_load_state_dict(net, os.path.join(checkpoint_path, str(fold), f'{net_name}.pt'))
    return nets

def save_checkpoint(fold, nets, net_names):
    for net, net_name in zip(nets, net_names):
        if net:
            torch.save(net.state_dict(), os.path.join(checkpoint_path, str(fold), f'{net_name}.pt'))

def init_net():
    if args['net']['type'] == 'CLAM_0':
        return net2devices(CLAM_0(), devices)
    elif args['net']['type'] == 'CLAM_1':
        return net2devices(CLAM_1(), devices)
    elif args['net']['type'] == 'CLAM_2':
        return net2devices(CLAM_2(), devices)
    elif args['net']['type'] == 'DTFD':
        dim_reduction = net2devices(DimReduction(1024, 512), devices)
        # dim_reduction = net2devices(DimReduction(768, 512), devices)
        attention = net2devices(Attention(512, 128, False), devices)
        classifier = net2devices(Classifier(512, 2), devices)
        net2 = net2devices(DTFD_T2(), devices)
        return dim_reduction, attention, classifier, net2

def init_optimizer(nets):
    if args['optimizer']['type'] == 'Adam':
        return Adam(itertools.chain(*[filter(lambda p: p.requires_grad, net.parameters()) for net in nets]), lr=args['optimizer']['lr'], weight_decay=args['optimizer']['weight_decay'])
    elif args['optimizer']['type'] == 'AdamW':
        return AdamW(itertools.chain(*[filter(lambda p: p.requires_grad, net.parameters()) for net in nets]), lr=args['optimizer']['lr'], betas=(0.9, 0.999), eps=1e-8, weight_decay=args['optimizer']['weight_decay'], amsgrad=False)

def init_lr_scheduler(optimizer):
    if args['lr_scheduler']['type'] == 'OneCycleLR':
        # lr_scheduler = OneCycleLR(optimizer=optimizer, max_lr=args['optimizer']['lr'] * args['lr_scheduler']['lr_times'], epochs=args['max_epoch'], steps_per_epoch=len(train_loader), pct_start=0.3, div_factor=20, final_div_factor=1000)
        return None
    elif args['lr_scheduler']['type'] == 'MultiStepLR':
        return MultiStepLR(optimizer, args['lr_scheduler']['milestones'], args['lr_scheduler']['gamma'])

def init_loss_fn():
    if args['net']['type'] == 'DTFD':
        return nn.CrossEntropyLoss(reduction='none')
    else:
        return nn.CrossEntropyLoss()

