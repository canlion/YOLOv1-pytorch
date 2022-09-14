import random

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn


COLORS = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
    (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
    (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
]  # https://sashamaps.net/docs/resources/20-colors/

CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


def init_seed(seed: int):
    """시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = True
    cudnn.deterministic = False


def draw_box(img, boxes, scores, cls_indices):
    size = min(img.shape[:2])
    thick = max(int(size * .01), 1)
    text_thick = max(thick//2, 1)
    scale = size * .0025
    for ltrb, score, cls_idx in zip(boxes, scores, cls_indices):
        l, t, r, b = (int(x) for x in ltrb)
        color = COLORS[int(cls_idx)][::-1]
        img = cv2.rectangle(img, (l, t), (r, b), color, thick)

    for ltrb, score, cls_idx in zip(boxes, scores, cls_indices):
        l, t, r, b = (int(x) for x in ltrb)
        color = COLORS[int(cls_idx)][::-1]
        cls_label = CLASSES[int(cls_idx)]
        (text_w, text_h), baseline = cv2.getTextSize(cls_label, cv2.FONT_HERSHEY_PLAIN, scale, text_thick)
        img = cv2.rectangle(img, (l, b-text_h-baseline), (l+text_w, b), color, -1)
        img = cv2.putText(img, cls_label, (l, b-baseline//2), cv2.FONT_HERSHEY_PLAIN, scale, (255, 255, 255), text_thick)

    return img