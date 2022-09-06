# TODO
# 1. 모델 생성
# 2. 데이터로더 생성
# 3. 학습 루프 정의
# 4. validation 루프 정의
# 5. 모델 저장
# 6. MLflow 이용

import sys
import argparse
import random
from typing import Callable
from collections import defaultdict

import colors
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from model import YOLOv1
from loss import YOLOv1Loss
from dataset import VOC, make_batch, Transform
from utils import init_seed


def main(args: argparse.Namespace):
    init_seed(args.seed)

    # TODO: model
    model = YOLOv1(args.backbone, args.inp_size, args.B, args.C)
    backbone_config = model.backbone_config

    # TODO: dataset
    data_mean, data_std = backbone_config['mean'], backbone_config['std']
    train_transform = Transform(inp_size=args.inp_size, mean=data_mean, std=data_std, is_train=True)
    voc_train = VOC(
        voc_root=args.voc_root,
        datasets=(('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val')),
        S=args.S,
        transform=train_transform
    )

    test_transform = Transform(inp_size=args.inp_size, mean=data_mean, std=data_std, is_train=False)
    voc_valid = VOC(
        voc_root=args.voc_root,
        datasets=(('2007test', 'test'),),
        S=args.S,
        transform=test_transform
    )

    batch_size = args.batch_size // args.subdivision

    train_loader = DataLoader(
        voc_train, batch_size, True,
        num_workers=args.num_worker,
        collate_fn=make_batch,
        drop_last=True,
    )
    valid_loader = DataLoader(
        voc_valid, batch_size, False,
        num_workers=args.num_worker,
        collate_fn=make_batch,
    )


    # TODO: optimizer, criterion
    # loss는 평균이 아닌 합으로 계산하므로 lr, decay를 배치사이즈에 맞게 조절
    lr = args.lr / args.batch_size
    w_decay = args.weight_decay * args.batch_size

    criterion = YOLOv1Loss(args.B, args.C, args.lambda_coord, args.lambda_noobj)

    weights = [p for n, p in model.named_parameters() if n.endswith('weight') and '.bn' not in n]
    biases = [p for n, p in model.named_parameters() if n.endswith('bias') or '.bn' in n]
    assert len(tuple(model.parameters())) == len(weights) + len(biases)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(weights, lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(weights, lr=lr, momentum=args.momentum, weight_decay=w_decay)
    optimizer.add_param_group({'params': biases, 'lr': lr, 'momentum': args.momentum})
    # optimizer = torch.optim.SGD(model.added_layers.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [75, 105], gamma=.1)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=.1)

    # def lr_lambda(epoch):
    #     if epoch < 10:
    #         return .1 * epoch
    #     elif epoch < 75:
    #         return 1.
    #     elif epoch < 105:
    #         return .1
    #     else:
    #         return .01
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # TODO: train & validation loop
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    warmup_step = 0
    warmup_epochs = args.warmup_epochs
    optimizer.zero_grad(set_to_none=True)
    num_steps = 0
    num_batches = 0
    gamma = 0.

    print(f'init lr: {lr}')
    # for epoch in range(args.epochs):
    epoch = 0
    while True:
        # 학습 단계
        model.train()
        epoch_loss = 0.
        losses_dict = defaultdict(float)
        tqdm_loader = tqdm(train_loader)
        for i, (img, target) in enumerate(tqdm_loader):
            # lr warm-up
            if epoch < warmup_epochs:
                for p_group in optimizer.param_groups:
                    p_group['lr'] = np.interp(
                        warmup_step,
                        [0, len(train_loader)*warmup_epochs],
                        [lr / 10, lr],
                    )
                warmup_step += 1

            img, target = img.to(device), target.to(device)

            pred = model(img)
            loss, losses = criterion(pred, target)

            # loss 출력
            tqdm_loader.set_description(' '.join(f'{key}: {val:.2f}' for key, val in losses.items()))
            epoch_loss += loss.item()
            for key, tensor in losses.items():
                losses_dict[key] += tensor.item()

            loss.backward()
            num_steps += 1

            # 그래디언트를 args.subdivision만큼 누적하여 업데이트
            if num_steps % args.subdivision == 0:
                optimizer.step()
                optimizer.zero_grad()
                num_batches += 1

                # 스텝 수에 따라 lr 조정
                if num_batches == 20000:
                    gamma = .1
                elif num_batches == 30000:
                    gamma = .1

                if gamma != 0.:
                    for p_group in optimizer.param_groups:
                        p_group['lr'] *= gamma
                    last_lr = optimizer.param_groups[0]['lr']
                    print(f'lr updated: {last_lr}')
                    gamma = 0.

                if num_batches == 40000:  # 학습 종료
                    torch.save(model.state_dict(), './yolo_last.pth')
                    print('done!')
                    sys.exit(0)

        report_str = colors.red('loss:') + f'{epoch_loss / batch_size / len(train_loader)}, ' \
            + ' '.join([colors.red(f'{key}:') + f'{val / len(train_loader):.4f}' for key, val in losses_dict.items()])
        print(colors.cyan(f'epoch {epoch + 1}'), report_str, f'num update: {num_batches}')

        # validation
        model.eval()
        valid_loss = 0.
        v_losses_dict = defaultdict(float)
        for img, target in valid_loader:
            img, target = img.to(device), target.to(device)

            with torch.no_grad():
                pred = model(img)
                loss, losses = criterion(pred, target)
            valid_loss += loss.item()
            for key, tensor in losses.items():
                v_losses_dict[key] += tensor.item()

        report_str = colors.blue('valid loss:') + f'{valid_loss / batch_size / len(valid_loader)}, ' \
            + ' '.join([colors.blue(f'{key}:') + f'{val / len(valid_loader):.4f}' for key, val in v_losses_dict.items()])
        # TODO: VOC mAP metric
        # v_loss = valid_loss / batch_size / len(valid_loader)
        # print(colors.blue(f'valid loss: {v_loss}'))
        print(report_str)

        torch.save(model.state_dict(), './temp_yolo.pth')
        epoch += 1


if __name__ == '__main__':
    np.set_printoptions(linewidth=800)
    torch.set_printoptions(linewidth=800)

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='manual seed')
    parser.add_argument('--backbone', type=str, default='resnet18', help='timm backbone')
    parser.add_argument('--inp_size', type=int, default=448, help='input size')
    parser.add_argument('--B', type=int, default=2, help='num grid')
    parser.add_argument('--C', type=int, default=20, help='num classes')
    parser.add_argument('--S', type=int, default=7, help='num grid')
    parser.add_argument('--voc_root', type=str, default='', help='VOC dataset root directory path')
    parser.add_argument('--batch_size', type=int, default=64, help='train batch size (valid batch_size: 2*batch_size)')
    parser.add_argument('--num_worker', type=int, default=8, help='dataloader num_worker')
    parser.add_argument('--lambda_coord', type=float, default=5., help='localization loss weight')
    parser.add_argument('--lambda_noobj', type=float, default=.5, help='no-object confidence score loss weight')
    parser.add_argument('--lr', type=float, default=1e-2, help='initial lr')
    parser.add_argument('--momentum', type=float, default=.9, help='optimizer momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    # parser.add_argument('--epochs', type=int, default=135, help='train epochs')
    parser.add_argument('--warmup_epochs', type=int, default=2, help='learning rate warmup epochs')
    # parser.add_argument('--max_grad_norm', type=float, default=10., help='gradient clipping')
    parser.add_argument('--subdivision', type=int, default=2, help='')
    # parser.add_argument('--amp', type=bool, action='store_true', help='use automatic mixed precision')
    args = parser.parse_args()

    main(args)
