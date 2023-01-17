import os, torch, cv2
from scipy.stats import rankdata
from math import floor, ceil
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

from yang.yang import read_svs, mkdir, rmdir, img_add_xml, dict2json, read_patch, dict2csv
from yang.dl import net_load_state_dict, net2devices, read_h5, read_label_file
from yang.net import CLAM_2

devices = ['cuda:1']

task = 'HE'

if task == 'IHC':
    exp_name = 'ResNet50_CLAM2_alpha0.4_rankdata'
    args = {
        'input': {
            'exp': 'ResNet50_CLAM_2',
            'fold': 1
        },
        'cmap': 'jet',
        'alpha': 0.4,
        'save_patch': {
            'neg_num': 8,
            'pos_num': 8
        },
        # 'save_patch': None,
        'draw_xml': True,
        'features': {
            'type': 'ResNet50',
            'use_xml': False,
        },
        'type': 'rankdata', # ['rankdata', 'normalize']
    }
elif task == 'HE':
    exp_name = 'ResNet50_CLAM2_alpha0.4_rankdata_xml'
    args = {
        'input': {
            'exp': 'ResNet50_CLAM_2',
            'fold': 1
        },
        'cmap': 'jet',
        'alpha': 0.4,
        'save_patch': {
            'neg_num': 8,
            'pos_num': 8
        },
        # 'save_patch': None,
        'draw_xml': True,
        'features': {
            'type': 'ResNet50',
            'use_xml': True,
        },
        'type': 'rankdata', # ['rankdata', 'normalize']
    }

def init_feature_path():
    if task == 'IHC':
        return os.path.join('/home/yangxuan/CLAM/features/IHC', args['features']['type'])
    elif task == 'HE':
        return os.path.join('/home/yangxuan/CLAM/features/HE', 'xml' if args['features']['use_xml'] else 'no_xml', args['features']['type'])

label_file_name = os.path.join('/home/yangxuan/dataset', task, 'label.csv')
svs_path = os.path.join('/home/yangxuan/dataset', task)
xml_path = os.path.join('/home/yangxuan/dataset', task + '_xml')

input_exp, input_fold = args['input']['exp'], args['input']['fold']
state_dict_name = os.path.join('/home/yangxuan/CLAM/checkpoint', task, input_exp, str(input_fold), 'net.pt')
label_train_file_name = os.path.join('/home/yangxuan/CLAM/split/', task, input_exp, f'train_{input_fold}.csv')
label_val_file_name = os.path.join('/home/yangxuan/CLAM/split/', task, input_exp, f'val_{input_fold}.csv')

feature_path = init_feature_path() 
out_dir = os.path.join('/home/yangxuan/CLAM/heatmap/', task, exp_name)
res_name = os.path.join(out_dir, 'res.csv')

rmdir(out_dir)
mkdir(out_dir)
# save args
dict2json(os.path.join(out_dir, 'arg.json'), args)

# scores(patch_num,) coords(patch_num, 2)
def gen_heatmap(svs_name, scores, coords, xml_name=None, out_name=None, alpha=0.5, cmap='coolwarm'):
    img, base = read_svs(svs_name, level=2, needs=['base'], mode='RGB')
    h, w, patch_size = img.shape[0], img.shape[1], ceil(256 / base)
    coords = np.ceil(coords / base).astype(int)

    heatmap = np.array(Image.new(size=(w, h), mode='RGB', color=(255, 255, 255))) # (h, w, 3)

    if args['type'] == 'rankdata':
        scores = rankdata(scores, 'average') / len(scores)
    elif args['type'] == 'normalize':
        scores = (scores - scores.min()) / (scores.max() - scores.min())

    cmap = plt.get_cmap(cmap)
    for score, coord in zip(scores, coords):
        j, i = coord
        block = np.full((min(h - i, patch_size), min(w - j, patch_size)), score)
        heatmap[i:i + patch_size, j:j + patch_size] = cmap(block, bytes=np.uint8)[:, :, :3]
    
    heatmap = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0, heatmap)

    if xml_name:
        heatmap = img_add_xml(heatmap, xml_name, base, color=(0, 255, 0), thickness=5)

    if out_name:
        plt.imsave(out_name, heatmap)

    return heatmap

def save_patch(svs_name, out_path, scores, coords):
    orders = scores.argsort()

    def func(out_path, is_pos=True):
        mkdir(out_path)
        file_names, sub_scores, sub_coords = [], [], []
        order = orders[-1:max(-len(orders), -args['save_patch']['pos_num']) - 1:-1] if is_pos else orders[:min(len(orders), args['save_patch']['neg_num'])]

        for i, idx in enumerate(order, 1):
            patch = read_patch(svs_name, coords[idx])
            sub_scores.append(scores[idx])
            sub_coords.append(coords[idx])
            plt.imsave(os.path.join(out_path, f'{i}_a{scores[idx]:.3f}_x{coords[idx][0]}_y{coords[idx][1]}.png'), patch)

        # dict2csv(os.path.join(out_path, 'res.csv'), {'score': sub_scores, 'coord': sub_coords})

    func(os.path.join(out_path, 'neg_patch'), False)
    func(os.path.join(out_path, 'pos_patch'), True)

def init_net():
    net = net2devices(CLAM_2(), devices)
    net_load_state_dict(net, state_dict_name)
    net.eval()

    return net

def get_scores(net, x):
    net.attention.softmax = False
    scores = net.attention(net.fc(x))
    net.attention.softmax = True
    return scores.detach().cpu().numpy()

if __name__ == '__main__':
    file_names, labels = read_label_file(label_file_name)
    # file_names, labels = file_names[:5], labels[:5]

    train_file_names, _ = read_label_file(label_train_file_name)
    val_file_names, _ = read_label_file(label_val_file_name)
    # file_names = ['2020-312872022-08-29_HE']

    net = init_net()

    probs, types = [], []
    for i, (file_name, label) in enumerate(zip(file_names, labels)):
        out_path = os.path.join(out_dir, 'tumor' if label else 'normal', file_name)
        svs_name = os.path.join(svs_path, file_name + '.svs')
        xml_name = os.path.join(xml_path, file_name + '.xml') if args['draw_xml'] else None
        out_name = os.path.join(out_path, file_name + '.png')
        h5_name = os.path.join(feature_path, file_name + '.h5')

        mkdir(out_path)

        x, coords = torch.from_numpy(read_h5(h5_name, 'features')).to(devices[0]), read_h5(h5_name, 'coords')

        scores = get_scores(net, x)
        logits = net(x)

        probs.append(torch.softmax(logits, dim=0)[label].item())
        if file_name in train_file_names:
            types.append('train')
        elif file_name in val_file_names:
            types.append('val')
        else:
            types.append('test')

        if args['save_patch']:
            save_patch(svs_name, out_path, scores, coords)

        heatmap = gen_heatmap(svs_name, scores, coords, xml_name, out_name, alpha=args['alpha'], cmap=args['cmap'])

        print(f'{i + 1}/{len(file_names)}, {file_name}')

    dict2csv(res_name, {'file_name': file_names.tolist(), 'label': labels.tolist(), 'prob': probs, 'type': types})
