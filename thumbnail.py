import os
from yang.yang import gen_thumbnail

if __name__ == '__main__':
    svs_dir = '/home/yangxuan/dataset/HE/'
    xml_dir = '/home/yangxuan/dataset/HE_xml/'
    out_dir = '.'

    file_names = ['B2020-292722022-08-15_HE']
    
    for i, file_name in enumerate(file_names):
        svs_name = os.path.join(svs_dir, file_name + '.svs')
        xml_name = os.path.join(xml_dir, file_name + '.xml')
        out_name = os.path.join(out_dir, file_name + '.png')

        gen_thumbnail(svs_name, xml_name, out_name)
