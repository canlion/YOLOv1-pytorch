import os
import argparse

import cv2
import torch
import numpy as np

from dataset import Transform
from model import YOLOv1
from utils import draw_box

# CLASSES = [
#     "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
#     "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
# ]


parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, help='')
parser.add_argument('--saved_pth', type=str, help='')
parser.add_argument('--gpu', action='store_true', help='')
parser.add_argument('--save_path', type=str, default='.', help='')
parser.add_argument('--score_threshold', type=float, default=.2, help='')
parser.add_argument('--iou_threshold', type=float, default=.4, help='')
parser.add_argument('--hflip', action='store_true', help='')

args = parser.parse_args()

# TODO: B, C 등의 모델 설정 적용
yolo = YOLOv1()
yolo.load_state_dict(torch.load(args.saved_pth, map_location=torch.device('cuda' if args.gpu else 'cpu')))
yolo.eval()

img = cv2.imread(args.img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
if args.hflip:
    img = cv2.flip(img, 1)

pred = yolo.predict(img, args.score_threshold, args.iou_threshold)
img = draw_box(img, *pred)

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
img_name = os.path.basename(args.img).split('.')[0]
cv2.imwrite(os.path.join(args.save_path, img_name + '_predict.jpg'), img)
