import os
from yang.yang import get_file_names_by_suffix, stitch

if __name__ == '__main__':

    # svs_dir = '/home/yangxuan/dataset/IHC/'
    # patch_dir = '/home/yangxuan/code/python/CLAM/results/IHC/patches'
    # out_dir = '/home/yangxuan/data/stitch/IHC/'
    svs_dir = '/home/yangxuan/dataset/IHC/'
    # patch_dir = '/home/yangxuan/CLAM/patches/HE_30/patches/'
    patch_dir = '/home/yangxuan/code/python/CLAM/results/IHC/patches_not_full_rect/'
    # out_dir = '/home/yangxuan/CLAM/stitch/HE_30/'
    out_dir = '/home/yangxuan/CLAM/tmp/'
    # file_names = get_file_names_by_suffix(patch_dir, '.h5')
    file_names = ['2020-312872022-08-29_IHC']

    false = 0
    for i, file_name in enumerate(file_names):
        try:
            svs_name = os.path.join(svs_dir, file_name + '.svs')
            patch_name = os.path.join(patch_dir, file_name + '.h5')
            out_name = os.path.join(out_dir, file_name + '.png')
            stitch(svs_name, patch_name, out_name, 3)
        except Exception as e:
            print(e)
            false += 1

        print(f'{i + 1}/{len(file_names)}')

    print(f'false:{false}')
