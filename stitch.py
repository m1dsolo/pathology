import os
from yang.yang import get_file_names_by_suffix, stitch, mkdir

svs_dir = os.path.join('/home/yangxuan/dataset/HE/')
patch_dir = os.path.join('/home/yangxuan/CLAM/patches/HE/HE/')
out_dir = os.path.join('/home/yangxuan/CLAM/stitch/HE/HE/')
# svs_dir = os.path.join('/home/yangxuan/dataset/HE/')
# patch_dir = os.path.join('/home/yangxuan/CLAM/patches/HE/HE1/')
# out_dir = os.path.join('/home/yangxuan/CLAM/stitch/HE/HE1/')
out_type = 'jpg'

if __name__ == '__main__':
    mkdir(out_dir)

    # file_names = get_file_names_by_suffix(patch_dir, '.h5')
    file_names = ['B2020-292722022-08-15_HE']

    err = 0
    for i, file_name in enumerate(file_names):
        svs_name = os.path.join(svs_dir, file_name + '.svs')
        patch_name = os.path.join(patch_dir, file_name + '.h5')
        out_name = os.path.join(out_dir, file_name + '.' + out_type)
        try:
            stitch(svs_name, patch_name, out_name, 2, thumbnail_size=960)
        except Exception as e:
            print(e)
            err += 1

        print(f'{i + 1}/{len(file_names)}')

    print(f'err:{err}')
