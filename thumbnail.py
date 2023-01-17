import os
from yang.yang import gen_thumbnail, get_file_names_by_suffix, mkdir


if __name__ == '__main__':
    svs_dir = '/home/yangxuan/dataset/HE/'
    xml_dir = '/home/yangxuan/dataset/xml/HE/'
    out_dir = '/home/yangxuan/CLAM/thumbnail/HE/HE/'
    # svs_dir = '/home/yangxuan/dataset/IHC/'
    # xml_dir = '/home/yangxuan/dataset/xml/IHC/'
    # out_dir = '/home/yangxuan/CLAM/thumbnail/IHC/IHC/'
    out_type = 'jpg'
    mkdir(out_dir)

    file_names = get_file_names_by_suffix(svs_dir, '.svs')
    # file_names = ['B2020-292722022-08-15_HE']
    
    err = 0
    for i, file_name in enumerate(file_names, 1):
        svs_name = os.path.join(svs_dir, file_name + '.svs')
        xml_name = os.path.join(xml_dir, file_name + '.xml')
        out_name = os.path.join(out_dir, file_name + '.' + out_type)

        try:
            gen_thumbnail(svs_name, xml_name, out_name, thumbnail_size=960)
        except Exception as e:
            err += 1
            print(e)

        print(f'{i}/{len(file_names)}')
    print(f'err:{err}')
