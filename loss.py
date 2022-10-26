from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


class YOLOv1Loss(nn.Module):
    """
    YOLOv1 loss

    Args:
        B (int): 그리드의 predictor 수
        C (int): 클래수 수
        lambda_coord (float): localization loss 가중치
        lambda_noobj (float): no-object predictor의 confidence loss 가중치
    """
    def __init__(self, B=2, C=20, lambda_coord=5., lambda_noobj=.5):
        super().__init__()
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        torch.set_printoptions(linewidth=800)

    def forward(self, pred, target) -> Tuple[torch.Tensor, Dict]:
        """
        loss 계산
        - 계산 방식은 주석 참조

        Args:
             pred (torch.Tensor): (batch, S, S, B * 5 + C)
             target (torch.Tensor): (N, 8)
                오브젝트의 batch index, grid y index, grid x index, YOLO style x, y, w, h, class index

        Returns:
            loss (torch.Tensor): loss 총합
            loss_dict (dict): 개별 loss를 담은 딕셔너리 (key: loss, xy, wh, conf, conf~, iou, cls)
        """
        # pred: (batch, S, S, B * 5 + C)
        # target: (N, 8)
        #   (batch idx, grid y idx, grid x idx, x, y, w, h, class idx)
        # print(pred)

        # TODO: conf loss, box loss, no-obj conf loss, cls loss

        ################################################################################################################
        device = pred.device

        batch, S, _, L = pred.size()
        N = target.size()[0]
        assert L == self.B * 5 + self.C, 'L'

        # obj grid indicator
        obj_grid_index = target[:, :3].long()
        obj_grid_mask = torch.zeros(batch, S, S, dtype=torch.bool, device=device)
        obj_grid_mask.index_put_(torch.split(obj_grid_index, 1, dim=-1), torch.tensor(True, device=device))
        assert obj_grid_mask.sum().item() == N, 'obj_grid_mask'

        # object grid - pred에서 오브젝트가 위치한 그리드의 예측값
        obj_grid_pred = pred[obj_grid_index[:, 0], obj_grid_index[:, 1], obj_grid_index[:, 2]]
        assert obj_grid_pred.size() == (N, L), 'obj_grid_pred'

        obj_box_pred, obj_cls_pred = torch.split(obj_grid_pred, (self.B * 5, self.C), -1)
        assert obj_box_pred.size() == (N, self.B * 5)
        assert obj_cls_pred.size() == (N, self.C)
        obj_box_pred = obj_box_pred.view(N, self.B, 5)

        # class prob. loss
        cls_target = F.one_hot(target[..., -1].long(), num_classes=self.C).float().to(device)
        loss_cls_prob = torch.square(cls_target - obj_cls_pred).sum() / batch

        # IoU - pred, target의 YOLO style xywh를 디코딩하여 IoU 계산
        with torch.no_grad():
            box_pred_iou = obj_box_pred[..., :-1].clone()
            box_pred_iou[..., :2] += target[..., 1:3].flip(dims=(1,)).unsqueeze(1).repeat(1, self.B, 1)
            box_pred_iou[..., :2] /= S
            box_pred_iou[..., 2:].square_()
            assert box_pred_iou.size() == (N, self.B, 4)
            box_target_iou = target[..., 3:-1].clone()
            box_target_iou[..., :2] += target[..., 1:3].flip(dims=(1,))
            box_target_iou[..., :2] /= S
            assert box_target_iou.size() == (N, 4)

            ious = IoU(box_pred_iou, box_target_iou.unsqueeze(1))
            assert ious.size() == (N, self.B)

        iou_max, iou_max_idx = torch.max(ious, dim=-1)
        assert iou_max.size() == (N,) and iou_max_idx.size() == (N,)

        # highest iou predictor indicator
        resp_mask = torch.zeros_like(ious, dtype=torch.bool, device=device)
        resp_mask.index_put_((torch.arange(N), iou_max_idx), torch.tensor(True, device=device))
        assert resp_mask.sum().item() == N

        # responsible predictor - 그리드별로 B개의 predictor 중 IoU가 가장 높은 predictor의 예측값
        resp_box_pred = obj_box_pred[resp_mask]
        assert resp_box_pred.size() == (N, 5)

        # box loss
        loss_box_xy = torch.square(target[..., 3:5] - resp_box_pred[..., :2]).sum() / batch
        #  identity activation 사용하므로 예측값이 음수가 나올 수 있음. 따라서 wh의 sqrt를 타겟으로 이용
        loss_box_wh = torch.square(target[..., 5:7].sqrt() - resp_box_pred[..., 2:4]).sum() / batch
        loss_box = loss_box_xy + loss_box_wh

        # object confidence loss
        loss_conf = torch.square(iou_max.detach() - resp_box_pred[..., -1]).sum() / batch

        # no-object confidence loss
        #   not responsible predictor
        not_resp_box_pred = obj_box_pred[~resp_mask]
        assert not_resp_box_pred.size() == (N, 5)
        loss_noobj_conf = torch.square(0. - not_resp_box_pred[..., -1]).sum() / batch
        #   no-object grid
        noobj_box_pred = pred[~obj_grid_mask][..., :-self.C].view(-1, self.B, 5)
        assert noobj_box_pred.size() == (batch * S * S - N, self.B, 5)
        loss_noobj_conf += torch.square(0. - noobj_box_pred[..., -1]).sum() / batch

        loss = self.lambda_coord * loss_box + loss_cls_prob + loss_conf + self.lambda_noobj * loss_noobj_conf
        loss_dict = {
            'loss': loss, 'xy': loss_box_xy, 'wh': loss_box_wh, 'cls': loss_cls_prob,
            'conf': loss_conf, 'conf~': loss_noobj_conf, 'iou': iou_max.mean()
        }

        return loss, loss_dict


def IoU(box0: torch.Tensor, box1: torch.Tensor) -> torch.Tensor:
    """IoU 계산"""
    # box0: (N, B, 4), box1: (N, 1, 4)
    # iou: (N, B)
    # box1 = box1.unsqueeze(1)  # (N, 1, 4)

    box_0_xy, box_0_wh = box0[..., :2], box0[..., 2:4]
    box_1_xy, box_1_wh = box1[..., :2], box1[..., 2:4]

    box_0_lt, box_0_rb = box_0_xy - box_0_wh / 2., box_0_xy + box_0_wh / 2.
    box_1_lt, box_1_rb = box_1_xy - box_1_wh / 2., box_1_xy + box_1_wh / 2.

    max_lt = torch.maximum(box_0_lt, box_1_lt)
    min_rb = torch.minimum(box_0_rb, box_1_rb)
    inter_wh = (min_rb - max_lt).clip(min=0)

    inter_area = torch.prod(inter_wh, dim=-1)

    box_0_area = torch.prod(box_0_wh, dim=-1)
    box_1_area = torch.prod(box_1_wh, dim=-1)
    union_area = box_0_area + box_1_area - inter_area

    return (inter_area / union_area + 1e-7).clip(0., 1.)


if __name__ == '__main__':
    from utils import init_seed

    init_seed(0)

    torch.set_printoptions(linewidth=800)
    target = torch.tensor(
        [[0., 5., 2., 0.88020833, 0.3194707, 0.72916667, 0.48015123, 1.],
         [0., 4., 3., 0.5, 0.22778828, 0.69270833, 0.74669187, 14.]]
    ).to('cuda')

    target_flip = torch.tensor(
        [[0., 4., 6.-3., 1.-0.5, 0.22778828, 0.69270833, 0.74669187, 14.],
         [0., 5., 6.-2., 1.-0.88020833, 0.3194707, 0.72916667, 0.48015123, 1.]]
    ).to('cuda')

    pred = torch.rand(1, 7, 7, 30).to('cuda')
    pred_flip = pred.flip(dims=(2,))
    pred_flip[..., 0] = 1. - pred_flip[..., 0]
    pred_flip[..., 5] = 1. - pred_flip[..., 5]

    loss = YOLOv1Loss()
    print(loss(pred, target))
    print(loss(pred_flip, target_flip))
    print(loss(pred, torch.zeros(0, 8).to('cuda')))
