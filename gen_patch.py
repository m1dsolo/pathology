import os, openslide, cv2
import numpy as np
from matplotlib import pyplot as plt
from math import ceil

from yang.yang import read_svs, thresholding
from yang.dl import save_h5

def gen_patch(svs_name, out_name, threshold=200, patch_size=256, ratio=0.5, close_size=None, median_size=None):
    img, base = read_svs(svs_name, 2, ['base'])
    step = ceil(patch_size / base)
    mask = ~thresholding(img, threshold)
    if close_size:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))
    if median_size:
        mask = cv2.medianBlur(mask, median_size)
    plt.imsave('mask.png', mask, cmap='gray')

    coords = []
    for i in range(0, img.shape[0] - step, step):
        for j in range(0, img.shape[1] - step, step):
            patch = mask[i:i + step, j: j + step]
            if patch.sum() / patch.size >= ratio:
                coords.append((j * round(base), i * round(base)))

    save_h5(out_name, 'coords', coords, {'name': os.path.basename(svs_name)[:-4], 'patch_size': patch_size, 'patch_level': 0})

if __name__ == '__main__':
    svs_path = '/home/yangxuan/dataset/HE/'
    out_path = '/home/yangxuan/CLAM/patches/HE/HE/'
    file_names = ['B2020-292722022-08-15_HE']

    for i, file_name in enumerate(file_names, 1):
        svs_name = os.path.join(svs_path, file_name + '.svs')
        out_name = os.path.join(out_path, file_name + '.h5')

        try:
            gen_patch(svs_name, out_name, 150, ratio=0.9, close_size=7, median_size=7)
        except Exception as e:
            print(e)

        print(f'{i}/{len(file_names)}')
