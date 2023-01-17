import torch, os
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter

from yang.yang import dict2json, update_json, mkdir
from yang.dl import FinalLogger, EarlyStop, set_seed, read_label_file, get_net_one_device, WithNone, Logger
from yang.dataset import WSI_Dataset, WSI_Loader

from args import *

def random_split(x: torch.Tensor, num):
    return torch.tensor_split(x[torch.randperm(x.shape[0])], num)

def get_cam(net: nn.Module, weighted_feature):
    return net.get_parameter('fc.weight') @ weighted_feature.T

def do_epoch_DTFD(is_train: bool, loader, dim_reduction, attention, classifier, net2, loss_fn, optimizer1=None, optimizer2=None, lr_scheduler=None):
    dim_reduction.train() if is_train else dim_reduction.eval()
    attention.train() if is_train else attention.eval()
    classifier.train() if is_train else classifier.eval()
    net2.train() if is_train else net2.eval()
    device = get_net_one_device(net2)

    with WithNone() if is_train else torch.no_grad():
        res_t1, res_t2 = Logger(), Logger()
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            full_reduct_feat = dim_reduction(x) # (patch_num, 512)
            full_attn_logits = attention(full_reduct_feat) # (patch_num,)

            sub_labels, sub_logits = [], []
            pbag_feats = []

            for idx in random_split(torch.arange(x.shape[0]), args['net']['pbag_num']):
                sub_x = x[idx]
                sub_labels.append(y)

                if is_train:
                    sub_reduct_feat = dim_reduction(sub_x) # (pinst_num, 512)
                    sub_attn_logits = attention(sub_reduct_feat) # (pinst_num,)
                else:
                    sub_reduct_feat = full_reduct_feat[idx] # (pinst_num, 512)
                    sub_attn_logits = full_attn_logits[idx] # (pinst_num,)

                sub_attn = torch.softmax(sub_attn_logits, dim=0) # (pinst_num,)
                sub_weighted_feat = sub_reduct_feat.mul(sub_attn.unsqueeze(dim=1)) #(pinst_num, 512)
                sub_tensor = sub_weighted_feat.sum(dim=0, keepdim=False) #(512,)
                sub_logit = classifier(sub_tensor) # (2,)

                sub_logits.append(sub_logit)

                patch_logits = get_cam(classifier, sub_weighted_feat).transpose(0, 1) # (pseudo_bag_num, 2)
                patch_probs = torch.softmax(patch_logits, dim=1) # (pseudo_bag_num, 2)

                if args['net']['distill'] == 'MaxS':
                    pbag_feats.append(sub_reduct_feat[patch_probs.argmax()])
                elif args['net']['distill'] == 'MaxMinS':
                    pbag_feats.append(sub_reduct_feat[torch.cat([patch_probs.argmax(), patch_probs.argmin()], dim=0)])
                elif args['net']['distill'] == 'AFS':
                    pbag_feats.append(sub_tensor)

            sub_logits = torch.stack(sub_logits) # (bag_num, 2)
            sub_labels = torch.tensor(sub_labels, dtype=torch.uint8, device=device) # (bag_num, )
            pbag_feats = torch.stack(pbag_feats) # (bag_num, 512) if not 'MaxMinS' else (bag_num * 2, 512)

            loss1 = loss_fn(sub_logits, sub_labels).mean()
            pbag_logits = net2(pbag_feats)
            loss2 = loss_fn(pbag_logits.unsqueeze(dim=0), y).mean()

            if is_train:
                optimizer1.zero_grad()
                loss1.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(dim_reduction.parameters(), args['net']['grad_clipping'])
                torch.nn.utils.clip_grad_norm_(attention.parameters(), args['net']['grad_clipping'])
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), args['net']['grad_clipping'])
                optimizer2.zero_grad()
                loss2.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(net2.parameters(), args['net']['grad_clipping'])

                optimizer1.step()
                optimizer2.step()
                if lr_scheduler:
                    lr_scheduler.step()

            pbag_probs = F.softmax(pbag_logits, dim=0)
            res_t1.add(loss1.item(), sub_labels, torch.softmax(sub_logits, dim=1)[:, 1])
            res_t2.add(loss2.item(), y.item(), pbag_probs[1].item())

            if (i + 1) % 10 == 0:
                print(f'batch: {i + 1}/{len(loader)}')

        return res_t1, res_t2

def train_DTFD():
    global train_label_file_name, test_label_file_name, feature_path, arg_path, args, checkpoint_path, seed, log_path, res_path, split_path
    global init_loss_fn, init_net, init_optimizer, init_lr_scheduler
    train_file_names, train_labels = read_label_file(train_label_file_name)
    global load_checkpoint, save_checkpoint
    test_file_names, test_labels = read_label_file(test_label_file_name)
    test_dataset = WSI_Dataset(test_file_names, test_labels, feature_path)
    test_loader = WSI_Loader(test_dataset, is_train=False)
    skf = StratifiedKFold(5)

    val_t1_logger = FinalLogger(['loss', 'acc', 'pos_acc', 'neg_acc', 'f1_score', 'auc'])
    val_t2_logger = FinalLogger(['loss', 'acc', 'pos_acc', 'neg_acc', 'f1_score', 'auc'])
    test_t1_logger = FinalLogger(['loss', 'acc', 'pos_acc', 'neg_acc', 'f1_score', 'auc'])
    test_t2_logger = FinalLogger(['loss', 'acc', 'pos_acc', 'neg_acc', 'f1_score', 'auc'])

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
        dim_reduction, attention, classifier, net2 = init_net()
        optimizer1, optimizer2 = init_optimizer([dim_reduction, attention, classifier]), init_optimizer([net2])
        lr_scheduler1, lr_scheduler2 = init_lr_scheduler(optimizer1), init_lr_scheduler(optimizer2)

        early_stop = EarlyStop(patience=20, min_stop_epoch=args['early_stop']['min_stop_epoch'], max_stop_epoch=args['early_stop']['max_stop_epoch'])

        epoch = 1
        while True:
            train_t1_res, train_t2_res = do_epoch_DTFD(True, train_loader, dim_reduction, attention, classifier, net2, loss_fn, optimizer1, optimizer2)
            val_t1_res, val_t2_res = do_epoch_DTFD(False, val_loader, dim_reduction, attention, classifier, net2, loss_fn)

            print(f'epoch:{epoch}')
            print(f'train_t1: {train_t1_res}')
            print(f'train_t2: {train_t2_res}')
            print(f'val_t1: {val_t1_res}')
            print(f'val_t2: {val_t2_res}')

            print(train_t2_res.matrix)
            print(val_t2_res.matrix)

            if writer:
                train_t1_res.to_writer(writer, 'train_t1/', ['loss', 'acc', 'pos_acc', 'neg_acc', 'f1_score', 'auc'], epoch)
                train_t2_res.to_writer(writer, 'train_t2/', ['loss', 'acc', 'pos_acc', 'neg_acc', 'f1_score', 'auc'], epoch)
                val_t1_res.to_writer(writer, 'val_t1/', ['loss', 'acc', 'pos_acc', 'neg_acc', 'f1_score', 'auc'], epoch)
                val_t2_res.to_writer(writer, 'val_t2/', ['loss', 'acc', 'pos_acc', 'neg_acc', 'f1_score', 'auc'], epoch)

            early_stop(val_t2_res.loss, epoch)

            if early_stop.best_epoch == epoch:
                save_checkpoint(fold, [dim_reduction, attention, classifier, net2, optimizer1, optimizer2, lr_scheduler1, lr_scheduler2], ['dim_reduction', 'attention', 'classifier', 'net2', 'optimizer1', 'optimizer2', 'lr_scheduler1', 'lr_scheduler2'])

            update_json(os.path.join(res_path, f'train_t1_res_{fold}.json'), {epoch: train_t1_res.to_dict(['loss', 'acc', 'pos_acc', 'neg_acc', 'f1_score', 'auc'])})
            update_json(os.path.join(res_path, f'train_t2_res_{fold}.json'), {epoch: train_t2_res.to_dict(['loss', 'acc', 'pos_acc', 'neg_acc', 'f1_score', 'auc'])})
            update_json(os.path.join(res_path, f'val_t1_res_{fold}.json'), {epoch: val_t1_res.to_dict(['loss', 'acc', 'pos_acc', 'neg_acc', 'f1_score', 'auc'])})
            update_json(os.path.join(res_path, f'val_t2_res_{fold}.json'), {epoch: val_t2_res.to_dict(['loss', 'acc', 'pos_acc', 'neg_acc', 'f1_score', 'auc'])})

            if args['lr_scheduler']['type'] == 'MultiStepLR':
                lr_scheduler1.step()
                lr_scheduler2.step()

            if early_stop.early_stop:
                print(f'stop at {epoch} epoch')
                break

            epoch += 1

        # checkpoint
        load_checkpoint(fold, [dim_reduction, attention, classifier, net2], ['dim_reduction', 'attention', 'classifier', 'net2'])
        val_t1_res, val_t2_res = do_epoch_DTFD(False, val_loader, dim_reduction, attention, classifier, net2, loss_fn)
        test_t1_res, test_t2_res = do_epoch_DTFD(False, test_loader, dim_reduction, attention, classifier, net2, loss_fn)

        # res
        val_t1_logger.update(val_t1_res, early_stop.stop_epoch, early_stop.best_epoch)
        val_t2_logger.update(val_t2_res, early_stop.stop_epoch, early_stop.best_epoch)
        test_t1_logger.update(test_t1_res, early_stop.stop_epoch, early_stop.best_epoch)
        test_t2_logger.update(test_t2_res, early_stop.stop_epoch, early_stop.best_epoch)

        # split
        train_dataset.to_csv(os.path.join(split_path, f'train_{fold}.csv'))
        val_dataset.to_csv(os.path.join(split_path, f'val_{fold}.csv'), val_t2_res.probs)
        test_dataset.to_csv(os.path.join(split_path, f'val_{fold}.csv'), test_t2_res.probs)

    dict2json(os.path.join(res_path, 'val_t1_res.json'), val_t1_logger.get_res_dict())
    dict2json(os.path.join(res_path, 'val_t2_res.json'), val_t2_logger.get_res_dict())
    dict2json(os.path.join(res_path, 'test_t1_res.json'), test_t1_logger.get_res_dict())
    dict2json(os.path.join(res_path, 'test_t2_res.json'), test_t2_logger.get_res_dict())
