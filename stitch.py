import os
from yang.yang import get_file_names_by_suffix, stitch, mkdir

if __name__ == '__main__':
    svs_dir = '/home/yangxuan/dataset/HE/'
    patch_dir = '/home/yangxuan/CLAM/patches/HE/HE_xml/'
    out_dir = '/home/yangxuan/CLAM/stitch/HE/HE_xml/'

    mkdir(out_dir)

    file_names = get_file_names_by_suffix(patch_dir, '.h5')
    # file_names = ['2021-386372022-08-29_HE']

    false = 0
    for i, file_name in enumerate(file_names):
        svs_name = os.path.join(svs_dir, file_name + '.svs')
        patch_name = os.path.join(patch_dir, file_name + '.h5')
        out_name = os.path.join(out_dir, file_name + '.png')
        try:
            stitch(svs_name, patch_name, out_name, 3)
        except Exception as e:
            print(e)
            false += 1
        # stitch(svs_name, patch_name, out_name, 3)

        print(f'{i + 1}/{len(file_names)}')

    print(f'false:{false}')
