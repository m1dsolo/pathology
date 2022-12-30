import os, cv2, openslide, h5py
import numpy as np
from xml.etree import ElementTree as ET
from PIL import Image
from matplotlib import pyplot as plt
from queue import Queue
from types import FunctionType
from matplotlib import pyplot as plt
import json

Image.MAX_IMAGE_PIXELS = None

def get_file_names_by_suffix(path, suffix):
    return [file[:-len(suffix)] for file in os.listdir(path) if file[-len(suffix):] == suffix]

# xml --> [[(x1, y1), (x2, y2), ...], [...], ...]
def xml2polygons(xml_name, base=1):
    tree = ET.parse(xml_name)
    root = tree.getroot()
    res = []
    for coordinate in root.iter('Coordinates'):
        arr = []
        for child in coordinate.iter('Coordinate'):
            arr.append((float(child.attrib['X']) * base, float(child.attrib['Y']) * base))
        res.append(np.array(arr, np.int32))
    return res

def xml2mask(xml_name, shape, base=1):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, xml2polygons(xml_name, base), 255)
    return mask

def add_polylines(thumbnail, xml_name, base=1):
    return cv2.polylines(thumbnail, xml2polygons(xml_name, base), True, (0, 0, 255), 3)

def gen_thumbnail(svs_name, xml_name, out_name=None, thumbnail_size=1920):
    with openslide.open_slide(svs_name) as slide:
        thumbnail = np.array(slide.get_thumbnail((thumbnail_size, thumbnail_size)))
        base = slide.dimensions[1] / thumbnail.shape[0]

    if xml_name:
        thumbnail = add_polylines(thumbnail, xml_name, 1 / base)

    if out_name:
        Image.fromarray(thumbnail).save(out_name)

    return thumbnail

def crop_thumbnail(svs_name, xml_name, png_name, out_name, thumbnail_size=1920):
    png = Image.open(png_name)
    with openslide.open_slide(svs_name) as slide:
        base = slide.dimensions[0] / png.size[0]

    rect = xml2polygons(xml_name, 1)[0].astype(np.int64)
    x0, x1, y0, y1 = int(1e9), -int(1e9), int(1e9), -int(1e9)
    for point in rect:
        x0 = min(x0, point[0] / base)
        x1 = max(x1, point[0] / base)
        y0 = min(y0, point[1] / base)
        y1 = max(y1, point[1] / base)

    png = png.crop((x0, y0, x1, y1))
    png.thumbnail((thumbnail_size, thumbnail_size))
    png.save(out_name)

# ------------------------------------------------------------------
# ------------------------------------------------------------------

# img[img >= threshold] = 255
def thresholding(img, threshold=None):
    if not threshold:
        val, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th, int(val) + 1
    return cv2.threshold(img, threshold - 1, 255, cv2.THRESH_BINARY)[1]

def get_edge(img):
    edge = []
    h, w = img.shape
    for i in range(h):
        for j in (range(0, w) if i in (0, h - 1) else (0, w - 1)):
            edge.append((i, j))
    return edge

def remove_hole(img):
    mask = np.full_like(img, 255, dtype=np.uint8)
    vis = np.zeros_like(img, dtype=bool)

    q = Queue()
    for i, j in get_edge(img):
        if not img[i, j]:
            q.put((i, j))
            vis[i, j] = True

    while q.qsize():
        for _ in range(q.qsize()):
            i, j = q.get()
            mask[i, j] = 0
            for a, b in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                ii, jj = i + a, j + b
                if ii >= 0 and ii < img.shape[0] and jj >= 0 and jj < img.shape[1] and not vis[ii, jj] and not img[ii, jj]:
                    q.put((ii, jj))
                    vis[ii, jj] = True

    return mask

# 生成去除空洞的且在mask内的二值化图像
def gen_mask(svs_name, xml_name, out_name=None, level=2, threshold=200):
    img, shape, base = read_svs(svs_name, level, ['shape', 'base'])
    mask = xml2mask(xml_name, img.shape, 1 / base) & ~thresholding(img, threshold)
    mask = remove_hole(mask)
    mask = cv2.resize(mask, shape)

    if out_name:
        cv2.imwrite(out_name, mask)
    return mask[::-1]

# ------------------------------------------------------------------
# ------------------------------------------------------------------

def rotate(img):
    return img[::-1].T

def calc_similarity(img1, img2):
    res = 0
    for _ in range(4):
        score = np.logical_and(img1, img2).sum() / np.logical_or(img1, img2).sum()
        res = max(res, score)
        img1 = rotate(img1)
    return res

# [(x1, y1, w1, h1), (x2, y2, w2, h2), ...]
def find_rects(img, ratio=0.005):
    th = ~thresholding(img, 250)
    area = img.shape[0] * img.shape[1]
    cnt, label, values, centroid = cv2.connectedComponentsWithStats(th, 4, cv2.CV_32S)

    rects = []
    for i in range(1, cnt):
        if values[i, 4] >= area * ratio:
            rects.append(values[i, 0:4])

    return rects

def find_target_rect(ihc, he, ratio=0.005, thumbnail_size=1080, ihc_threshold=235, he_threshold=200):
    rects = find_rects(ihc, ratio)
    tgt = thresholding(he, he_threshold)
    tgt = cv2.resize(tgt, (thumbnail_size, thumbnail_size))
    scores = []
    for rect in rects:
        part = ihc[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        src = thresholding(part, ihc_threshold)
        src = cv2.resize(src, (thumbnail_size, thumbnail_size))
        scores.append(calc_similarity(tgt, src))

    return rects[scores.index(max(scores))]

def rect2points(rect, base=1, pad=0):
    j, i, w, h = rect
    i0, j0, i1, j1 = i * base - pad, j * base - pad, (i + h) * base + pad, (j + w) * base + pad
    return [(i0, j0), (i1, j0), (i1, j1), (i0, j1)]

def indent_xml(elem, level=0):
    i = '\n' + level * '\t'
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + '\t'
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent_xml(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def rect2xml(rect, base, pad=0):
    asap = ET.Element('ASAP_Annotations')
    annotations = ET.Element('Annotations')
    annotation = ET.Element('Annotation', {'Name': 'rect', 'Type': 'Rectangle', 'PartOfGroup': 'None', 'Color': '#F4FA58'})
    coordinates = ET.Element('Coordinates')
    for i, point in enumerate(rect2points(rect, base, pad)):
        coordinates.append(ET.Element('Coordinate', {'Order': str(i), 'X': str(point[1]), 'Y': str(point[0])}))

    annotation.append(coordinates)
    annotations.append(annotation)
    asap.append(annotations)

    indent_xml(asap)

    return ET.ElementTree(asap)

def read_svs(svs_name, level=2, needs=None):
    def process_needs(needs):
        res = []
        for need in needs:
            if need == 'shape':
                res.append(slide.dimensions)
            elif need == 'base':
                res.append(slide.level_downsamples[level])
        return res

    with openslide.open_slide(svs_name) as slide:
        img = np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]).convert('L'))

        return (img, *process_needs(needs)) if needs else img

"""
描述：在IHC图像中找到与HE图像相匹配的矩形区域并返回其对应的xml文件。

level: 读取svs文件的图像金字塔层数
ratio: 只统计面积大于ratio的矩形
thumbnail_size: 算法过程中中间缩略图的边长
pad: 返回的矩形区域的padding
ihc_threshold: 算法过程中IHC图像二值化时所用的阈值
he_threshold: 算法过程中HE图像二值化时所用的阈值

return: 矩形的四个点坐标
"""
def gen_rect_xml(ihc_name, he_name, out_name=None, level=2, ratio=0.005, thumbnail_size=1080, pad=0, ihc_threshold=235, he_threshold=200):
    (ihc, base), he = read_svs(ihc_name, level, ['base']), read_svs(he_name, level)
    rect = find_target_rect(ihc, he, ratio, thumbnail_size, ihc_threshold, he_threshold)

    if out_name:
        rect2xml(rect, base, pad).write(out_name)

    return rect2points(rect, base, pad)

# ------------------------------------------------------------------
# ------------------------------------------------------------------

def write_list(file_name, lst):
    if lst:
        with open(file_name, 'w') as f:
            for line in lst:
                f.write(f'{line}\n')

def count(img):
    cnt, _ = np.histogram(img.flatten(), bins=np.arange(257), density=False)
    return {key: val for key, val in zip(np.arange(257), cnt) if val != 0}

def gen_merge_fig(img1, img2, out_name=None, no_tick=False):
    if isinstance(img1, str):
        img1 = cv2.imread(img1)
    if isinstance(img2, str):
        img2 = cv2.imread(img2)
    fig, axes = plt.subplots(1, 2)

    if no_tick:
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

    axes[0].imshow(img1)
    axes[1].imshow(img2)

    plt.tight_layout(pad=0)
    if out_name:
        fig.savefig(out_name, dpi=600)
        plt.close(fig)

def unzip_list(lst):
    return lst if len(lst) > 1 else lst[0]

# 获取前cnt大的连通块
def get_white_masks(img, cnt=1):
    _, label, values, _ = cv2.connectedComponentsWithStats(img, 4, cv2.CV_32S)
    areas = values[1:, 4]
    idxs = areas.argsort()[len(areas) - 1:len(areas) - 1 - cnt:-1] + 1

    return unzip_list([(label == idx).astype(np.uint8) * 255 for idx in idxs])

def show_images(imgs, out_name=None, title=None, cmap='viridis', no_tick=False):
    h, w = 1, 1
    if isinstance(imgs, list):
        if isinstance(imgs[0], list):
            h, w = len(imgs), max([len(row) for row in imgs])
        else:
            w = len(imgs)

    fig, axes = plt.subplots(h, w, squeeze=False)
    gen_title = title() if isinstance(title, FunctionType) else None
    for i in range(h):
        for j in range(len(imgs[i])):
            axes[i, j].imshow(imgs[i][j], cmap=cmap)
            if no_tick:
                axes[i, j].xaxis.set_ticks([])
                axes[i, j].yaxis.set_ticks([])
            if title:
                axes[i, j].set_title(next(gen_title) if gen_title else title[i][j])

    plt.tight_layout(pad=0)

    if out_name:
        plt.savefig(out_name)
        plt.close()

    return fig

# 1d --> 2d
def reshape_list(lst, shape):
    assert len(lst) == shape[0] * shape[1]
    return [lst[i * shape[1]:(i + 1) * shape[1]] for i in range(shape[0])]

def point_is_in_rect(point, rect):
    x, y = point
    x_min, x_max, y_min, y_max = min(rect[:, 0]), max(rect[:, 0]), min(rect[:, 1]), max(rect[:, 1])
    return x >= x_min and x <= x_max and y >= y_min and y <= y_max

def stitch(svs_name, patch_name, out_name=None, level=2):
    with h5py.File(patch_name, 'r') as f:
        dset = f['coords']

        with openslide.open_slide(svs_name) as slide:
            patch_size = int(dset.attrs['patch_size'] * slide.level_downsamples[dset.attrs['patch_level']])
            down_samples = slide.level_downsamples[level]
            patch_size = int(np.ceil(patch_size / down_samples))
            w, h = slide.level_dimensions[level]
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

            for coord in dset[:]:
                patch = np.array(slide.read_region(tuple(coord), level, (patch_size, patch_size)).convert('RGB'))

                coord = (int(np.ceil(coord[0] / down_samples)), int(np.ceil(coord[1] / down_samples)))
                size = canvas[coord[1]:coord[1] + patch_size, coord[0]:coord[0] + patch_size, :].shape[:2]
                canvas[coord[1]:coord[1] + patch_size, coord[0]:coord[0] + patch_size, :] = patch[:size[0], :size[1], :]

    if out_name:
        Image.fromarray(canvas).save(out_name)

    return canvas

def process_h5(patch_name, xml_name, out_name):
    rect = xml2polygons(xml_name)[0]
    x0, x1, y0, y1 = rect[0][0], rect[2][0], rect[0][1], rect[2][1]
    rect = np.array([[x0, y0], [x0, y1 - 256], [x1 - 256, y1 - 256], [x1 - 256, y0]])

    coords = []
    # features = []
    fin = h5py.File(patch_name, 'r')

    for coord in fin['coords']:
        if point_is_in_rect(coord, rect):
            coords.append(coord)
            # features.append(feature)

    fout = h5py.File(out_name, 'w') 
    fout.create_dataset('/coords', data=coords)
    fout['coords'].attrs.update(fin['coords'].attrs)
    # fout['features'] = features

    print(f'{len(coords)}/{len(fin["coords"])}')

    fin.close(), fout.close()

def dict2json(json_name, d):
    with open(json_name, 'w') as f:
        f.write(json.dumps(d, indent=2, separators=(', ', ': ')))

def json2dict(json_name):
    if not os.path.exists(json_name):
        return {}
    with open(json_name, 'r') as f:
        return json.load(f)

def update_json(json_name, d):
    json_dict = json2dict(json_name)
    json_dict.update(d)
    dict2json(json_name, json_dict)

def mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
