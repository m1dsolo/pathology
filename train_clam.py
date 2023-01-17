import torch, os
from torch.nn import functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter

from yang.yang import dict2json, update_json, mkdir
from yang.dl import FinalLogger, EarlyStop, set_seed, read_label_file, get_net_one_device, WithNone, Logger
from yang.dataset import WSI_Dataset, WSI_Loader

from args import *

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

def train_CLAM():
    global train_label_file_name, test_label_file_name, feature_path, arg_path, args, checkpoint_path, seed, log_path, res_path, split_path
    global init_loss_fn, init_net, init_optimizer, init_lr_scheduler
    train_file_names, train_labels = read_label_file(train_label_file_name)
    global load_checkpoint, save_checkpoint

    train_file_names, train_labels = read_label_file(train_label_file_name)
    test_file_names, test_labels = read_label_file(test_label_file_name)
    test_dataset = WSI_Dataset(test_file_names, test_labels, feature_path)
    test_loader = WSI_Loader(test_dataset, is_train=False)

    skf = StratifiedKFold(5)

    val_logger = FinalLogger(['loss', 'acc', 'pos_acc', 'neg_acc', 'f1_score', 'auc'])
    test_logger = FinalLogger(['loss', 'acc', 'pos_acc', 'neg_acc', 'f1_score', 'auc'])

    for fold, (train_i, val_i) in enumerate(skf.split(train_file_names, train_labels), 1):
        mkdir(os.path.join(checkpoint_path, str(fold)))

        # seed
        set_seed(seed)

        # writer
        writer_path = os.path.join(log_path, str(fold))
        mkdir(writer_path)
        writer = SummaryWriter(writer_path)

        # Dataset DataLoader
        train_dataset = WSI_Dataset(train_file_names[train_i], train_labels[train_i], feature_path)
        train_loader = WSI_Loader(train_dataset, is_train=True)
        val_dataset = WSI_Dataset(train_file_names[val_i], train_labels[val_i], feature_path)
        val_loader = WSI_Loader(val_dataset, is_train=False)

        loss_fn = init_loss_fn()
        net = init_net()
        optimizer = init_optimizer([net])
        lr_scheduler = init_lr_scheduler(optimizer)

        early_stop = EarlyStop(patience=20, min_stop_epoch=args['early_stop']['min_stop_epoch'], max_stop_epoch=args['early_stop']['max_stop_epoch'])

        epoch = 1
        while True:
            train_res = do_epoch(True, train_loader, net, loss_fn, optimizer)
            val_res = do_epoch(False, val_loader, net, loss_fn)

            print(f'epoch:{epoch}')
            print(f'train: {train_res}')
            print(f'val: {val_res}')

            print(train_res.matrix)
            print(val_res.matrix)

            if writer:
                train_res.to_writer(writer, 'train/', ['loss', 'acc', 'pos_acc', 'neg_acc', 'f1_score', 'auc'], epoch)
                val_res.to_writer(writer, 'val/', ['loss', 'acc', 'pos_acc', 'neg_acc', 'f1_score', 'auc'], epoch)

            early_stop(val_res.loss, epoch)

            if early_stop.best_epoch == epoch:
                save_checkpoint(fold, [net, optimizer, lr_scheduler], ['net', 'optimizer', 'lr_scheduler'])

            update_json(os.path.join(res_path, f'train_res_{fold}.json'), {epoch: train_res.to_dict(['loss', 'acc', 'pos_acc', 'neg_acc', 'f1_score', 'auc'])})
            update_json(os.path.join(res_path, f'val_res_{fold}.json'), {epoch: val_res.to_dict(['loss', 'acc', 'pos_acc', 'neg_acc', 'f1_score', 'auc'])})

            if early_stop.early_stop:
                print(f'stop at {epoch} epoch')
                break

            epoch += 1

        # checkpoint
        load_checkpoint(fold, [net], ['net'])
        val_res = do_epoch(False, val_loader, net, loss_fn)
        test_res = do_epoch(False, test_loader, net, loss_fn)

        # res
        val_logger.update(val_res, early_stop.stop_epoch, early_stop.best_epoch)
        test_logger.update(test_res, early_stop.stop_epoch, early_stop.best_epoch)

        # split
        train_dataset.to_csv(os.path.join(split_path, f'train_{fold}.csv'))
        val_dataset.to_csv(os.path.join(split_path, f'val_{fold}.csv'), val_res.probs)
        test_dataset.to_csv(os.path.join(split_path, f'test_{fold}.csv'), test_res.probs)

        # writer
        if writer:
            test_res.to_writer(writer, 'test/', ['loss', 'acc', 'pos_acc', 'neg_acc', 'f1_score', 'auc'], fold)

    dict2json(os.path.join(res_path, 'val_res.json'), val_logger.get_res_dict())
    dict2json(os.path.join(res_path, 'test_res.json'), test_logger.get_res_dict())
