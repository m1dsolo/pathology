import os, torch, cv2
from scipy.stats import rankdata
from math import floor, ceil
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

from yang.yang import read_svs, mkdir, img_add_xml, dict2json, read_patch
from yang.dl import net_load_state_dict, net2devices, read_h5, read_label_file
from yang.net import CLAM_2

devices = ['cuda:0']

label_file_name = '/home/yangxuan/dataset/HE/label.csv'
label_train_file_name = '/home/yangxuan/dataset/HE/label_train.csv'
svs_path = '/home/yangxuan/dataset/HE/'
xml_path = '/home/yangxuan/dataset/HE_xml/'
feature_path = '/home/yangxuan/CLAM/features/HE/HE/'
out_dir = '/home/yangxuan/CLAM/heatmap/HE/coolwarm_alpha1_rankdata_xml/'
res_name = os.path.join(out_dir, 'res.json')

args = {
    'cmap': 'coolwarm',
    'alpha': 1,
    'save_patch': {
        'neg_num': 8,
        'pos_num': 8
    },
    # 'save_patch': None,
    'add_xml': True
}

# scores(patch_num,) coords(patch_num, 2)
def gen_heatmap(svs_name, scores, coords, xml_name=None, out_name=None, alpha=0.5, cmap='coolwarm'):
    img, base = read_svs(svs_name, level=2, needs=['base'], mode='RGB')
    h, w, patch_size = img.shape[0], img.shape[1], ceil(256 / base)
    coords = np.ceil(coords / base).astype(int)

    heatmap = np.array(Image.new(size=(w, h), mode='RGB', color=(255, 255, 255))) # (h, w, 3)

    # scores = (scores - scores.min()) / (scores.max() - scores.min())
    scores = rankdata(scores, 'average') / len(scores)

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

def save_patch(out_path, scores, coords):
    neg_path = os.path.join(out_path, 'neg_patch')
    pos_path = os.path.join(out_path, 'pos_patch')
    mkdir(neg_path), mkdir(pos_path)

    order = scores.argsort()
    neg_scores, pos_scores = [], []
    for i, idx in enumerate(order[:min(len(order), args['save_patch']['neg_num'])], 1):
        patch = read_patch(svs_name, coords[idx])
        neg_scores.append(scores[idx])
        plt.imsave(os.path.join(neg_path, f'{i}.png'), patch)
    for i, idx in enumerate(order[-1:max(-len(order), -args['save_patch']['pos_num']) - 1:-1], 1):
        patch = read_patch(svs_name, coords[idx])
        pos_scores.append(scores[idx])
        plt.imsave(os.path.join(pos_path, f'{i}.png'), patch)
    dict2json(os.path.join(neg_path, 'scores.json'), {f'{i + 1}.png': float(neg_scores[i]) for i in range(len(neg_scores))})
    dict2json(os.path.join(pos_path, 'scores.json'), {f'{i + 1}.png': float(pos_scores[i]) for i in range(len(pos_scores))})

def init_net():
    net = net2devices(CLAM_2(), devices)
    net_load_state_dict(net, '/home/yangxuan/CLAM/checkpoint/IHC/clam_2/net.pt')
    net.attention.softmax = False
    net.eval()

    return net

def clam_2(net, x):
    logits = net(x)
    scores = net.attention(net.fc(x))
    return scores.detach().cpu().numpy(), logits

if __name__ == '__main__':
    file_names, labels = read_label_file(label_file_name)
    train_file_names, _ = read_label_file(label_train_file_name)
    # file_names = ['2020-312872022-08-29_HE']

    net = init_net()

    probs = []
    for i, (file_name, label) in enumerate(zip(file_names, labels)):
        out_path = os.path.join(out_dir, 'tumor' if label else 'normal', file_name)
        svs_name = os.path.join(svs_path, file_name + '.svs')
        xml_name = os.path.join(xml_path, file_name + '.xml') if args['add_xml'] else None
        out_name = os.path.join(out_path, file_name + '.png')
        h5_name = os.path.join(feature_path, file_name + '.h5')

        mkdir(out_path)

        x, coords = torch.from_numpy(read_h5(h5_name, 'features')).to(devices[0]), read_h5(h5_name, 'coords')

        scores, logits = clam_2(net, x)
        probs.append(torch.softmax(logits, dim=0)[1].item())

        if args['save_patch']:
            save_patch(out_path, scores, coords)

        heatmap = gen_heatmap(svs_name, scores, coords, xml_name, out_name, alpha=args['alpha'], cmap=args['cmap'])

        print(f'{i + 1}/{len(file_names)}, {file_name}')

    d = {'file_name': file_names.tolist(), 'label': labels.tolist(), 'prob': probs, 'in_train': [int(file_name in train_file_names) for file_name in file_names]}
    dict2json(res_name, d)
