import os
import xml.etree.ElementTree as elemtree


VOC_PATH = '/mnt/hdd/datasets/voc'
SETS = (('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007test', 'test'))
CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


def ltrb_to_xywh_norm(img_size, l, t, r, b):
    """ltrb 포맷을 이미지의 가로, 세로 사이즈로 정규화된 xywh 포맷으로 변환"""
    img_h, img_w = img_size
    x = (l + r) / 2. - 1.
    y = (t + b) / 2. - 1.
    w = r - l
    h = b - t
    x, y, w, h = x/img_w, y/img_h, w/img_w, h/img_h
    return x, y, w, h


# VOC 데이터셋의 xml 어노테이션을 텍스트 파일로 변환
for year, dataset in SETS:
    dataset_root = os.path.join(VOC_PATH, f'VOCdevkit/VOC{year}')
    sample_set_path = os.path.join(dataset_root, f'ImageSets/Main/{dataset}.txt')
    anno_dir_path = os.path.join(dataset_root, 'Annotations')
    cnt = 0

    print(f'convert {year}-{dataset} annotation xml to txt')
    sample_set = open(sample_set_path, 'r')
    sample_ids = (sample_id.strip() for sample_id in sample_set.readlines())
    for sample_id in sample_ids:
        anno_path = os.path.join(anno_dir_path, f'{sample_id}.xml')
        tree = elemtree.parse(anno_path)
        root = tree.getroot()
        size = root.find('size')
        h, w = int(size.find('height').text), int(size.find('width').text)

        empty = True
        label_txt = open(os.path.join(anno_dir_path, f'{sample_id}.txt'), 'w')
        for obj in root.iter('object'):
            difficult = int(obj.find('difficult').text)
            cls = obj.find('name').text
            if (cls not in CLASSES) or difficult == 1:
                continue
            cls_id = CLASSES.index(cls)
            xmlbox = obj.find('bndbox')
            ltrb = [float(xmlbox.find(key).text) for key in ('xmin', 'ymin', 'xmax', 'ymax')]
            xywh = ltrb_to_xywh_norm((h, w), *ltrb)
            label_txt.write(' '.join(str(x) for x in (*xywh, cls_id)) + '\n')
            empty = False
        label_txt.close()
        if empty:
            raise ValueError(f'empty file: {anno_path}')
        cnt += 1
    print(f'done: {year}-{dataset} total {cnt} files')
    sample_set.close()





