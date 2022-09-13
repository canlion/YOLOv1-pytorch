import os
import random
from typing import Tuple, Callable, Optional
from itertools import chain

import cv2
import torch
from torch.utils.data import Dataset
import numpy as np


CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


class VOC(Dataset):
    """
    PASCAL VOC 데이터셋 오브젝트

    이미지와 어노테이션은 반환. augmentation을 위한 callable 객체가 주어지면 augmentation을 수행.
    {voc_root}/VOCdevkit/VOC{year}/ImageSets/Main/{imageset}.txt에 포함된 샘플들을 순회.

    Args:
        voc_root (str): VOC 데이터셋 디렉토리 경로
        datasets (tuple): ({year}, {imageset}) 쌍을 담은 튜플, ex) (('2007', 'train'), ('2012', 'train'), ...)
        S (int): YOLO 그리드 분할 수
        transform (Callable): augmentation 객체
    """
    def __init__(self, voc_root: str, datasets: Tuple[Tuple[str, str], ...], S=7, transform: Optional[Callable] = None):
        super().__init__()
        assert os.path.exists(voc_root), f'존재하지 않는 디렉토리: {voc_root}'
        self.transform = transform
        self.S = S

        # 데이터 샘플 경로 수집
        all_ids = []
        for year, ds in datasets:
            ds_root = os.path.join(voc_root, f'VOCdevkit/VOC{year}')
            sample_ids_path = os.path.join(ds_root, f'ImageSets/Main/{ds}.txt')
            with open(sample_ids_path, 'r') as f:
                ids = [(year, sample_id.strip()) for sample_id in f.readlines()]
                assert len(ids) != 0, f'빈 데이터셋: {sample_ids_path}'
            all_ids.append(ids)

        self.all_ids = list(chain(*all_ids))
        # print(self.all_ids[:5])
        self.img_path_form = os.path.join(voc_root, 'VOCdevkit/VOC{year}/JPEGImages/{sample_id}.jpg')
        self.ann_path_form = os.path.join(voc_root, 'VOCdevkit/VOC{year}/Annotations/{sample_id}.txt')

    def __len__(self) -> int:
        return len(self.all_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        주어진 인덱스에 해당하는 데이터 샘플(이미지 & 어노테이션) 반환.

        이미지: ch first
        어노테이션: (N, 7)
            오브젝트가 위치한 그리드 y, x 인덱스와 YOLO 스타일의 정규화된 x, y, w, h, 클래스 라벨
            오브젝트가 없는 경우 (0, 7) 텐서 반환.

        Args:
            idx (int): 데이터 샘플 인덱스

        Returns:
            image (torch.Tensor)
            annotations (torch.Tensor)
        """
        # 인덱스에 해당하는 데이터 샘플 로드
        year, sample_id = self.all_ids[idx]
        img_path = self.img_path_form.format(year=year, sample_id=sample_id)
        ann_path = self.ann_path_form.format(year=year, sample_id=sample_id)
        # print(img_path)

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        ann = np.loadtxt(ann_path, ndmin=2)

        if self.transform:  # 어그멘테이션
            transformed = self.transform(img, ann)
            img, ann = transformed['img'], transformed['box']
            # transformed = self.transform(image=img, bboxes=ann)
            # img, ann = transformed['image'], np.array(transformed['bboxes'])

        img = np.transpose(img, (2, 0, 1))  # ch first

        if ann.shape[0]:
            # 하나의 그리드에 다수의 오브젝트가 포함된 경우 셔플하여 무작위로 오브젝트 선택
            ann = ann[np.random.permutation(ann.shape[0])]
            ann_idx = np.floor(ann[:, :2] * self.S)
            uniq_ann_idx, uniq_idx = np.unique(ann_idx, return_index=True, axis=0)

            ann = ann[uniq_idx]
            ann[:, :2] = ann[:, :2] * self.S - uniq_ann_idx  # img xy -> grid xy
            # ann[:, 2:4] = np.sqrt(ann[:, 2:4])
            ann = np.concatenate([uniq_ann_idx[..., ::-1], ann], axis=-1)  # 그리드 인덱스와 YOLO style xywhc 결합
            return torch.tensor(img), torch.tensor(ann)
        return torch.tensor(img), torch.zeros(0, 7)

    def aug_test(self, idx: int, save_dir_path: str, sample_id: Optional[str] = None):
        """
        어그멘테이션 객체 테스트

        인덱스에 해당하는 데이터 샘플 원본과 어그멘테이션 후 데이터 샘플을 시각화하여 이미지로 저장함.

        Args:
            idx (int): 데이터 샘플 인덱스 지정
            save_dir_path (str): 이미지 저장 경로 -> {save_dir_path}/original.jpg & {save_dir_path}/aug.jpg
            sample_id (str): 특정 샘플 id 지정
        """
        if sample_id is not None:
            idx = [x[1] for x in self.all_ids].index(sample_id)

        year, sample_id = self.all_ids[idx]

        img_path = self.img_path_form.format(year=year, sample_id=sample_id)
        ann_path = self.ann_path_form.format(year=year, sample_id=sample_id)

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        ann = np.loadtxt(ann_path)
        if ann.ndim == 1:
            ann = ann[np.newaxis]
        # ann[:, 2:4] = np.sqrt(ann[:, 2:4])
        self.save_sample(img, ann, os.path.join(save_dir_path, 'original.jpg'))

        sample = self[idx]
        img, ann = sample[0].numpy(), sample[1].numpy()
        print(ann)

        if self.transform is not None:
            img = ((np.transpose(img, (1, 2, 0)) * self.transform.std + self.transform.mean) * 255.).astype(np.uint8)
        grid_xy = ann[:, 1::-1]
        xywh = ann[:, 2:]
        xywh[:, :2] += grid_xy
        xywh[:, :2] /= self.S

        self.save_sample(img, xywh, os.path.join(save_dir_path, 'aug.jpg'), draw_grid=True)

    def save_sample(self, img, ann, img_path, draw_grid=False):
        img_h, img_w = img.shape[:2]
        ann_ = ann.copy()
        # ann_[:, 2:4] = np.square(ann_[:, 2:4])

        ann_[:, :4] *= np.array([[img_w, img_h, img_w, img_h]])

        ltrbc = np.stack([
            ann_[:, 0] - ann_[:, 2] / 2.,
            ann_[:, 1] - ann_[:, 3] / 2.,
            ann_[:, 0] + ann_[:, 2] / 2.,
            ann_[:, 1] + ann_[:, 3] / 2.,
            ann[..., -1],
        ], axis=-1)
        # print(ltrb)
        for row in ltrbc:
            l, t, r, b, c = (int(v) for v in row)
            img = cv2.line(img, (l, b), (r, t), (128, 0, 255), 1)
            img = cv2.rectangle(img, (l, t), (r, b), (128, 0, 255), 2)
            img = cv2.putText(img, CLASSES[c], (l, b), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 0, 128), 2)

        for x, y in ann_[:, :2]:
            img = cv2.circle(img, (int(x), int(y)), 5, (255, 128, 0), -1)

        if draw_grid:
            for grid_x in np.arange(0, img.shape[1], img.shape[1]//self.S):
                img = cv2.line(img, (grid_x, 0), (grid_x, img.shape[0]), (0, 128, 0), 1)
            for grid_y in np.arange(0, img.shape[0]+1, img.shape[0]//self.S):
                img = cv2.line(img, (0, grid_y), (img.shape[0], grid_y), (0, 128, 0), 1)

        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def make_batch(samples):
    """
    dataloader collate fn

    이미지와 어노테이션 배치 생성.
    어노테이션에 배치 인덱스를 추가함.
        (N, 8) / batch index, grid y index, grid x index, yolo style x, y, w, h, class
    """
    def concat_batch_idx(t, batch_idx):
        batch_idx_col = torch.ones(t.shape[0], 1).fill_(batch_idx)
        return torch.cat([batch_idx_col, t], dim=-1)
    imgs = [sample[0] for sample in samples]
    anns = [sample[1] for sample in samples if sample[1].size()[0] != 0]

    img_batch = torch.stack(imgs, dim=0)
    if anns:
        ann_batch = torch.cat([concat_batch_idx(ann, i) for i, ann in enumerate(anns)], dim=0)
    else:
        ann_batch = torch.zeros((0, 8))

    return img_batch.float(), ann_batch.float()


def rand_scale(s: float) -> float:
    """darknet 방식의 random scaling factor 생성"""
    scale = random.uniform(1., s)
    if random.random() >= .5:
        return scale
    else:
        return 1. / scale


class Transform:
    """
    YOLOv1 augmentation
    """
    def __init__(self,
                 jitter=.2,
                 exposure=1.5,
                 saturation=1.5,
                 inp_size=448,
                 obj_size_threshold=.01,
                 mean=None,
                 std=None,
                 is_train=True,
                 ):
        self.jitter = jitter
        self.exposure = exposure
        self.saturation = saturation
        self.inp_size = inp_size
        self.mean = mean
        self.std = std
        self.is_train = is_train
        self.eps = 1e-3
        self.obj_size_threshold = obj_size_threshold

    def saturate_exposure_img(self, img):
        """
        RGB -> HSV -> S, V scaling -> RGB
        """
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        s, e = rand_scale(self.saturation), rand_scale(self.exposure)
        img_hsv[..., 1] *= s
        img_hsv[..., 2] *= e
        img_hsv[..., 1:].clip(0, 255, out=img_hsv[..., 1:])
        img_rgb = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return img_rgb

    def random_crop_resize(self, img, xywhc=None):
        """
        random crop & pad -> resize
        """
        img_h, img_w = img.shape[:2]

        jitter_h, jitter_w = int(img_h * self.jitter), int(img_w * self.jitter)
        crop_l = int((2 * random.random() - 1) * jitter_w)
        crop_t = int((2 * random.random() - 1) * jitter_h)
        crop_r = int((2 * random.random() - 1) * jitter_w)
        crop_b = int((2 * random.random() - 1) * jitter_h)

        img_crop = img[max(crop_t, 0): img_h-crop_b, max(crop_l, 0): img_w-crop_r]
        pad_l, pad_t, pad_r, pad_b = (max(0, -val) for val in (crop_l, crop_t, crop_r, crop_b))
        img_crop = cv2.copyMakeBorder(img_crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)

        if xywhc is not None:
            x, y, w, h = np.split(xywhc[:, :4], 4, axis=-1)
            l = x - w/2.
            t = y - h/2.
            r = x + w/2.
            b = y + h/2.

            img_crop_h, img_crop_w = img_crop.shape[:2]
            y_scale, x_scale = img_h / img_crop_h, img_w / img_crop_w
            y_shift, x_shift = crop_t / img_crop_h, crop_l / img_crop_w

            l = (l * x_scale - x_shift).clip(0., 1.)
            t = (t * y_scale - y_shift).clip(0., 1.)
            r = (r * x_scale - x_shift).clip(0., 1.)
            b = (b * y_scale - y_shift).clip(0., 1.)

            xywhc = np.concatenate([
                (l+r)/2., (t+b)/2., r-l, b-t, xywhc[:, -1:]
            ], axis=-1)

            return img_crop, xywhc
        return img_crop, None

    def scale_pad_img(self, img):
        img_h, img_w = img.shape[:2]
        if self.is_train:
            dw, dh = self.jitter * img_w, self.jitter * img_h
            new_ar = (img_w + random.uniform(-dw, dw)) / (img_h + random.uniform(-dh, dh))
        else:
            # dw, dh = 0, 0
            new_ar = img_w / img_h

        if new_ar < 1.:
            nh = self.inp_size
            nw = int(nh * new_ar)
        else:
            nw = self.inp_size
            nh = int(nw / new_ar)

        if self.is_train:
            dx, dy = random.randint(0, self.inp_size - nw), random.randint(0, self.inp_size - nh)
        else:
            dx, dy = 0, 0

        img_resize = cv2.resize(img, (nw, nh))
        img_pad = cv2.copyMakeBorder(img_resize, dy, self.inp_size - nh - dy, dx, self.inp_size - nw - dx,
                                     cv2.BORDER_CONSTANT, value=0)
        return img_pad, (nw / self.inp_size, nh / self.inp_size), (dx / self.inp_size, dy / self.inp_size)

    def flip_horizontal(self, img, xywhc=None):
        if self.is_train and random.random() >= .5:
            img = cv2.flip(img, 1)
            xywhc[:, 0] = 1. - xywhc[:, 0]
        return img, xywhc

    def resize(self, img, h, w):
        img_resize = cv2.resize(img, (w, h))
        return (img_resize / 255. - self.mean) / self.std

    def __call__(self, img, xywhc=None):
        if self.is_train:
            img, xywhc = self.random_crop_resize(img, xywhc)
            img = self.saturate_exposure_img(img)
            img, xywhc = self.flip_horizontal(img, xywhc)
            xywhc[:, :4] = xywhc[:, :4].clip(min=0., max=1.)

            if self.obj_size_threshold:
                w_valid = xywhc[:, 2] >= self.obj_size_threshold
                h_valid = xywhc[:, 3] >= self.obj_size_threshold
                xywhc = xywhc[np.logical_and(w_valid, h_valid)]

        img = cv2.resize(img, (self.inp_size, self.inp_size))

        if self.mean is not None and self.std is not None:
            img = (img / 255. - self.mean) / self.std

        return {'img': img, 'box': xywhc}


if __name__ == '__main__':
    import argparse
    from collections import defaultdict
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, default=0, help='')
    parser.add_argument('--count_obj_cls', action='store_true', help='')
    parser.add_argument('--sample_id', type=str, default=None, help='')
    args = parser.parse_args()

    np.set_printoptions(linewidth=800)
    torch.set_printoptions(linewidth=800)

    transform = Transform(mean=(.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), is_train=True)

    ds = VOC(
        voc_root='/mnt/hdd/datasets/voc',
        datasets=(('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val')),
        # datasets=(('2007test', 'test'),),
        transform=transform
    )

    if args.count_obj_cls:
        num_obj_per_cls = defaultdict(int)

        for i in tqdm(range(len(ds))):
            sample = ds[i]
            # if sample[1] is not None:
            annotations = sample[1][..., -1].numpy().astype(np.int32)
            # print(annotations)
            for cls_idx in annotations:
                num_obj_per_cls[int(cls_idx)] += 1

        for idx in range(20):
            print(f'{CLASSES[idx]}: {num_obj_per_cls[idx]}')

    print('len(ds):', len(ds))
    sample = ds[args.idx]
    # print('img shape:', sample[0].shape)
    # print('ann shape:', sample[1].shape)
    # print(sample[1])
    print('augmentation test')
    ds.aug_test(args.idx, '.', args.sample_id)
    #
    # dl = torch.utils.data.DataLoader(ds, 64, True, collate_fn=make_batch)
    # batch = next(iter(dl))
    # print(batch[0].size(), batch[1].size())
    # print(batch[1])
    # print('##########################################')
    # train_loader = torch.utils.data.DataLoader(
    #     ds, 4, True,
    #     num_workers=2,
    #     collate_fn=make_batch,
    #     drop_last=True,
    # )
    #
    # print(next(iter(train_loader))[1])