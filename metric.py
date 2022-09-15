from collections import defaultdict
from typing import Optional, List

import numpy as np


class VOCEvaluator:
    def __init__(self, iou_thres: float = .5, num_classes: int = 20, classes: Optional[List] = None):
        self.iou_thres = iou_thres
        self.num_classes = num_classes
        self.eval_category = defaultdict(lambda: {'match': [], 'scores': [], 'num_gt': 0, 'AP': 0.})
        self.gt_classes = list(range(num_classes)) if classes is None else classes
        assert len(self.gt_classes) == num_classes, 'num_classes != len(classes)'

    def add(self, dt: np.ndarray, gt: np.ndarray):
        # dt(detection): l, t, r, b, class idx, score
        # gt(ground-truth): l, t, r, b, class idx

        for c_idx in range(self.num_classes):  # 카테고리별로 진행
            dt_, gt_ = np.zeros((0,)), np.zeros((0,))
            if dt.size:
                dt_ = dt[dt[..., 4].astype(np.int32) == c_idx]
            if gt.size:
                gt_ = gt[gt[..., 4].astype(np.int32) == c_idx]
                self.eval_category[c_idx]['num_gt'] += gt_.shape[0]

            if dt_.size:
                dt_match = np.zeros((dt_.shape[0],))
                dt_match_gt_idx = np.zeros_like(dt_match)
                dt_ = dt_[np.argsort(dt_[..., -1])[::-1]]  # score 기준 정렬
                ious = self.iou_matrix(dt_[:, :4], gt_[:, :4])

                if np.prod(ious.shape):
                    for dt_idx, iou_row in enumerate(ious):
                        max_iou = np.max(iou_row)
                        if max_iou >= self.iou_thres:
                            gt_idx = int(np.argmax(iou_row))
                            dt_match[dt_idx] = 1.
                            ious[:, gt_idx] = -1.
                            dt_match_gt_idx[dt_idx] = gt_idx

                self.eval_category[c_idx]['scores'].append(dt_[..., -1])
                self.eval_category[c_idx]['match'].append(dt_match)

    def accumulate(self):
        for c_idx in range(self.num_classes):
            # print('\n', self.gt_classes[c_idx], '-------------------')
            match, num_gt = self.eval_category[c_idx]['match'], self.eval_category[c_idx]['num_gt']
            scores = self.eval_category[c_idx]['scores']

            if num_gt and len(match):
                # print(match)
                scores = np.concatenate(scores, axis=0)
                sort_idx = np.argsort(scores)[::-1]
                match = np.concatenate(match, axis=0)[sort_idx]
                match_cumsum = np.cumsum(match)
                precision = match_cumsum / np.arange(1, match.shape[0]+1)
                recall = match_cumsum / num_gt

                # print('num_gt', num_gt)
                # print('num_dt', match.shape[0])
                # print('P')
                # print([f'{p:.2f}' for p in precision])
                # print('R')
                # print([f'{r:.2f}' for r in recall])
                # print(f'tp: {match.sum()}, fp: {match.shape[0] - match.sum()}')

                for i in range(len(precision)-1, 0, -1):
                    if precision[i] > precision[i-1]:
                        precision[i-1] = precision[i]

                ap = 0.
                ap += precision[0] * recall[0]
                for i in range(1, len(precision)):
                    pr = precision[i]
                    d_rc = recall[i] - recall[i-1]
                    ap += pr * d_rc

                self.eval_category[c_idx]['AP'] = ap

    def result(self):
        sum_AP = 0.
        ap_list = []
        for c_idx in range(self.num_classes):
            ap = self.eval_category[c_idx]['AP']
            sum_AP += ap
            ap_list.append((self.gt_classes[c_idx], ap))
            # print(f'class {str(self.gt_classes[c_idx]):20s} AP: {ap:.2f}')
        ap_list.sort(key=lambda x: x[0])
        for cls, ap in ap_list:
            print(f'class {cls:20s} AP: {ap * 100:.2f}')
        print(f'mAP: {sum_AP / self.num_classes * 100:.2f}')
        return sum_AP / self.num_classes

    def iou_matrix(self, dt, gt):
        # dt, gt: l, t, r, b
        # dt: (m, 4), gt: (n, 4)
        # iou: (m, n)

        dt, gt = dt[:, None], gt[None]  # (m, 1, 4), (1, n, 4)

        dt_lt, dt_rb = dt[..., :2], dt[..., 2:4]
        gt_lt, gt_rb = gt[..., :2], gt[..., 2:4]
        dt_wh = dt_rb - dt_lt + 1
        gt_wh = gt_rb - gt_lt + 1

        max_lt = np.maximum(dt_lt, gt_lt)
        min_rb = np.minimum(dt_rb, gt_rb)

        inter_wh = (min_rb - max_lt + 1).clip(min=0.)
        inter_area = np.prod(inter_wh, axis=-1)

        dt_area = np.prod(dt_wh, axis=-1)
        gt_area = np.prod(gt_wh, axis=-1)
        union_area = dt_area + gt_area - inter_area

        return inter_area / (union_area + np.spacing(1.))


if __name__ == '__main__':
    import os
    from glob import glob

    ##############################################
    # reference: https://github.com/Cartucho/mAP #
    ##############################################

    # 위의 reference repository에서 예제 파일을 pull하여 테스트
    dt_dir_path = '../od/mAP/input/detection-results'
    gt_dir_path = '../od/mAP/input/ground-truth'

    gt_classes = sorted([
        'bowl', 'chair', 'diningtable', 'door', 'doll', 'cabinetry', 'shelf', 'coffeetable', 'book',
        'windowblind', 'pottedplant', 'bookcase', 'countertop', 'tap', 'wastecontainer', 'bottle', 'tincan',
        'bed', 'pictureframe', 'nightstand', 'heater', 'pillow', 'backpack', 'tvmonitor', 'sofa', 'cup',
        'remote', 'vase', 'person', 'sink'
    ])

    def cvt_classes(x):
        if x not in gt_classes:
            gt_classes.append(x)
        return gt_classes.index(x)

    evaluator = VOCEvaluator(num_classes=len(gt_classes), classes=gt_classes)

    dt_list = glob(os.path.join(dt_dir_path, '*.txt'))
    for dt_path in dt_list:
        if os.path.getsize(dt_path):
            dt = np.loadtxt(dt_path, converters={0: cvt_classes}, ndmin=2)
            dt = dt[..., np.array([2, 3, 4, 5, 0, 1])]
        else:
            dt = np.zeros((0,))

        gt_path = os.path.join(gt_dir_path, os.path.basename(dt_path))
        if os.path.getsize(gt_path):
            gt = np.loadtxt(gt_path, converters={0: cvt_classes}, ndmin=2)
            gt = gt[..., np.array([1, 2, 3, 4, 0])]
        else:
            gt = np.zeros((0,))
        evaluator.add(dt, gt)
    evaluator.accumulate()
    evaluator.result()

    # reference와 동일한 결과 확인
