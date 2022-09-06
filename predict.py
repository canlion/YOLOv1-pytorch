import os
import argparse

import cv2
import torch
import numpy as np
import torchvision.ops as ops

from dataset import Transform
from model import YOLOv1


CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, help='')
parser.add_argument('--saved_pth', type=str, help='')
parser.add_argument('--gpu', action='store_true', help='')
parser.add_argument('--save_path', type=str, default='.', help='')
parser.add_argument('--threshold', type=float, default=.2, help='')
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

config = yolo.backbone_config
transform = Transform(mean=config['mean'], std=config['std'], inp_size=yolo.inp_size, is_train=False)

inp = transform(img)['img']
inp = torch.tensor(np.transpose(inp, (2, 0, 1))).unsqueeze(0).float()

out = yolo(inp)
_, S, _, _ = out.size()

out_ = out.clone().detach().numpy()[0]

img_h, img_w = img.shape[:2]
# for grid_x in np.arange(0, inp_copy.shape[1], inp_copy.shape[1] // 7):
#     inp_copy = cv2.line(inp_copy, (grid_x, 0), (grid_x, inp_copy.shape[0]), (0, 128, 0), 1)
# for grid_y in np.arange(0, inp_copy.shape[0] + 1, inp_copy.shape[0] // 7):
#     img = cv2.line(inp_copy, (0, grid_y), (inp_copy.shape[0], grid_y), (0, 128, 0), 1)
# for i in range(49):

for y_idx, row in enumerate(out_):
    for x_idx, col in enumerate(row):
        #     y_idx, x_idx = i // 7, i % 7
        #         boxes = out_[y_idx, x_idx, :10].reshape(2, 5)
        boxes = col[:10].reshape(2, 5)
        cls_scores = out_[y_idx, x_idx, -20:]
        for box in boxes:
            x, y, w, h, score = box
            scores = score * cls_scores
            x = ((x + x_idx) / S) * img_w
            y = ((y + y_idx) / S) * img_h
            w = (w ** 2) * img_w
            h = (h ** 2) * img_h

            l = int(x - w / 2.)
            t = int(y - h / 2.)
            r = int(x + w / 2.)
            b = int(y + h / 2.)

            l = min(max(l, 0), img_w)
            r = min(max(r, 0), img_w)
            t = min(max(t, 0), img_h)
            b = min(max(b, 0), img_h)

            # TODO: NMS는 현재 오브젝트들의 상대적 위치에 따른 에러 해결 후 적용
            cls_idx = np.argmax(scores)
            if scores[cls_idx] >= args.threshold:
                # print(box, x, y, w, h)
                # print(' '.join([f'{CLASSES[idx]}: {s:.4f} /' for idx, s in enumerate(scores)]))
                # print(colors.red(CLASSES[cls_idx]), scores[cls_idx])
                # print(l, t, r, b, w, h)
                # print()
                # TODO: 클래스별 박스 컬러 설정, 클래스명을 담는 박스 영역 설정, 이미지 크기에 따라 유동적인 폰트 스케일
                img = cv2.rectangle(img, (l, t), (r, b), (128, 0, 255), 2)
                img = cv2.circle(img, (int(x), int(y)), 5, (128, 0, 0), -1)
                img = cv2.putText(img, f'{CLASSES[cls_idx]}:{y_idx},{x_idx}', (l, b - 10),
                                       cv2.FONT_HERSHEY_DUPLEX, .75, (255, 0, 128), 1)

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
img_name = os.path.basename(args.img).split('.')[0]
cv2.imwrite(os.path.join(args.save_path, img_name + '_predict.jpg'), img)
