from typing import Tuple

import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from timm.data import resolve_data_config


class YOLOv1(nn.Module):
    def __init__(self, backbone='resnet18', inp_size=448, B=2, C=20):
        super().__init__()
        assert timm.is_model(backbone), f'timm: undefined model name: {backbone}'
        assert inp_size % 64 == 0, f'inp_size must be a multiple of 64: {inp_size}'

        self.B, self.C = B, C
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool='')
        self.backbone_config = resolve_data_config({}, model=self.backbone)
        self.mean, self.std = self.backbone_config['mean'], self.backbone_config['std']
        self.inp_size = inp_size
        # print(self.backbone.feature_info)
        #
        # for p in self.backbone.parameters():
        #     p.requires_grad = False

        feat_ch = self.backbone.feature_info[-1]['num_chs']
        feat_size = inp_size // 64
        len_pred = B * 5 + C

        self.added_layers = nn.Sequential(
            # original
            nn.Conv2d(feat_ch, 1024, 3, 1, padding=1),
            nn.LeakyReLU(.1, True),
            nn.Conv2d(1024, 1024, 3, 2, padding=1),
            nn.LeakyReLU(.1, True),
            nn.Conv2d(1024, 1024, 3, 1, padding=1),
            nn.LeakyReLU(.1, True),
            nn.Conv2d(1024, 1024, 3, 1, padding=1),
            nn.LeakyReLU(.1, True),
            nn.Flatten(),
            nn.Linear(feat_size ** 2 * 1024, 4096),
            nn.LeakyReLU(.1, True),
            nn.Dropout(p=.5),
            nn.Linear(4096, (feat_size ** 2) * len_pred),
            nn.Unflatten(1, (feat_size, feat_size, len_pred))
        )
        self.init_weights()

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.added_layers(x)
        # x = torch.permute(x, (0, 2, 3, 1))
        return x

    def init_weights(self):
        for l in self.added_layers:
            if isinstance(l, nn.Conv2d):
                # 다크넷 YOLOv1 가중치 초기화
                scale = torch.sqrt(torch.tensor(2./(l.in_channels * (l.kernel_size[0] ** 2))))
                l.weight.data = 2 * scale * torch.rand(l.weight.size()) - scale
                torch.nn.init.zeros_(l.bias.data)
            elif isinstance(l, nn.Linear):
                # 다크넷 YOLOv1 가중치 초기화
                scale = torch.sqrt(torch.tensor(2./l.in_features))
                l.weight.data = 2 * scale * torch.rand(l.weight.size()) - scale
                torch.nn.init.zeros_(l.bias.data)

    def normalization_config(self):
        return {key: self.backbone_config[key] for key in ['mean', 'std']}

    def predict(self, img: np.ndarray, score_threshold: float = .2, iou_threshold: float = .4)\
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        prediction + NMS

        Args:
             img (np.ndarray): HWC 3-dim RGB image
             score_threshold (float): prediction score threshold
             iou_threshold (float): NMS IoU threshold

        Returns:
            boxes (np.ndarray / N, 4)
            scores (np.ndarray / N,)
            class index (np.ndarray / N,)
        """
        def xywh2ltrb(xywh):
            x, y, w, h = np.split(xywh, 4, -1)
            l = x - w/2.
            t = y - h/2.
            r = x + w/2.
            b = y + h/2.
            return np.concatenate([l, t, r, b], axis=-1)

        from torchvision.ops import nms
        from dataset import Transform
        t = Transform(mean=self.mean, std=self.std, inp_size=self.inp_size, is_train=False)

        img_h, img_w = img.shape[:2]

        inp = t(img)['img']
        inp = torch.tensor(np.transpose(inp, (2, 0, 1))).float()

        pred = self(inp[None]).detach().numpy()[0]  # (S, S, L)
        S = pred.shape[0]
        boxes = pred[..., :self.B*5].reshape(S, S, self.B, 5)

        # box decoding
        grid_xy = np.meshgrid(np.arange(S), np.arange(S))
        grid_xy = np.transpose(np.stack(grid_xy, axis=0), (1, 2, 0))
        boxes[..., :2] += grid_xy[:, :, np.newaxis]
        boxes[..., :2] /= S
        boxes[..., 2:4] = np.square(boxes[..., 2:4])

        # confidence * class score
        scores = pred[..., np.newaxis, -self.C:] * boxes[..., -1, np.newaxis]  # (S, S, self.B, self.C)
        scores[scores < score_threshold] = 0.

        # nms
        boxes = boxes[..., :-1].reshape(-1, 4)
        boxes *= [[img_w, img_h, img_w, img_h]]
        boxes = xywh2ltrb(boxes)
        scores = scores.reshape(-1, self.C)

        for c_idx in range(self.C):
            score = scores[:, c_idx]
            nms_idx = nms(torch.tensor(boxes), torch.tensor(score), iou_threshold)
            score[~np.isin(np.arange(score.shape[0]), nms_idx)] = 0.

        scores_max = np.max(scores, axis=-1)
        scores_argmax = np.argmax(scores, axis=-1)
        indicator = scores_max > 0.

        return boxes[indicator].clip(0, [img_w, img_h, img_w, img_h]), scores_max[indicator], scores_argmax[indicator]


if __name__ == '__main__':
    from torchsummary import torchsummary

    yolo = YOLOv1(backbone='resnet18')
    fake_inp = torch.rand(1, 3, 448, 448)
    print(yolo(fake_inp).size())
    assert tuple(yolo(torch.rand(1, 3, 448, 448)).size()) == (1, 7, 7, 30)

    torchsummary.summary(yolo, fake_inp)
