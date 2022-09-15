import os
import argparse
import xml.etree.ElementTree as elemtree

import cv2
import numpy as np
import torch
from tqdm import tqdm

from model import YOLOv1
from metric import VOCEvaluator


CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


def load_ann(ann_path):
    tree = elemtree.parse(ann_path)
    root = tree.getroot()

    obj_list = []

    for obj in root.iter('object'):
        difficult = int(obj.find('difficult').text)
        cls = obj.find('name').text
        if (cls not in CLASSES) or difficult == 1:
            continue
        cls_id = CLASSES.index(cls)
        xmlbox = obj.find('bndbox')
        ltrb = [float(xmlbox.find(key).text) for key in ('xmin', 'ymin', 'xmax', 'ymax')]
        obj_list.append(ltrb+[cls_id])

    if obj_list:
        return np.array(obj_list)
    return np.zeros((0,))


def evaluate(args):

    device = 'cuda' if torch.cuda.is_available() and args.gpu else 'cpu'
    # print(torch.device('cuda' if args.gpu else 'cpu'))
    yolo = YOLOv1().to(device)
    yolo.load_state_dict(torch.load(args.saved_pth, map_location=torch.device('cuda' if args.gpu else 'cpu')))
    # print(next(yolo.parameters()).device)
    yolo.eval()

    evaluator = VOCEvaluator(num_classes=len(CLASSES), classes=CLASSES)

    ds_root = os.path.join(args.voc_root, f'VOCdevkit/VOC{args.year}')
    sample_ids_path = os.path.join(ds_root, f'ImageSets/Main/{args.ds}.txt')
    with open(sample_ids_path, 'r') as f:
        ids = [(args.year, sample_id.strip()) for sample_id in f.readlines()]

    img_path_form = os.path.join(args.voc_root, 'VOCdevkit/VOC{year}/JPEGImages/{sample_id}.jpg')
    ann_path_form = os.path.join(args.voc_root, 'VOCdevkit/VOC{year}/Annotations/{sample_id}.xml')

    for year, sample_id in tqdm(ids):
        img_path = img_path_form.format(year=year, sample_id=sample_id)
        ann_path = ann_path_form.format(year=year, sample_id=sample_id)

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        gt = load_ann(ann_path)

        boxes, scores, cls_indices = yolo.predict(img, device=device)
        if boxes.shape[0]:
            dt = np.concatenate([boxes, cls_indices[:, None], scores[:, None]], axis=-1)
        else:
            dt = np.zeros((0,))

        evaluator.add(dt, gt)

    evaluator.accumulate()
    evaluator.result()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--voc_root', type=str, help='')
    parser.add_argument('--year', type=str, help='')
    parser.add_argument('--ds', type=str, help='')
    parser.add_argument('--saved_pth', type=str, help='')
    parser.add_argument('--gpu', action='store_true', help='')

    args = parser.parse_args()

    evaluate(args)
