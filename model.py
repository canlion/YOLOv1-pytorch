import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.data import resolve_data_config


class YOLOv1(nn.Module):
    def __init__(self, backbone='resnet18', inp_size=448, B=2, C=20):
        super().__init__()
        assert timm.is_model(backbone), f'timm: undefined model name: {backbone}'
        assert inp_size % 64 == 0, f'inp_size must be a multiple of 64: {inp_size}'

        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool='')
        self.backbone_config = resolve_data_config({}, model=self.backbone)
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

            # test - fully conv
            # nn.Conv2d(feat_ch, 1024, 3, 1, padding=1),
            # nn.BatchNorm2d(1024),
            # nn.LeakyReLU(.1, True),
            # nn.Conv2d(1024, 1024, 3, 2, padding=1),
            # nn.BatchNorm2d(1024),
            # nn.LeakyReLU(.1, True),
            # nn.Conv2d(1024, 1024, 3, 1, padding=1),
            # nn.BatchNorm2d(1024),
            # nn.LeakyReLU(.1, True),
            # nn.Conv2d(1024, 1024, 3, 1, padding=1),
            # nn.BatchNorm2d(1024),
            # nn.LeakyReLU(.1, True),
            # nn.Conv2d(1024, 512, 1, 1),
            # nn.BatchNorm2d(512),
            # nn.LeakyReLU(.1, True),
            # nn.Conv2d(512, len_pred, 1, 1),
            # Rearrange('b (h w l) -> b h w l', h=feat_size, w=feat_size, l=len_pred)
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
                # torch.nn.init.kaiming_normal_(l.weight.data)
                # torch.nn.init.xavier_normal_(l.weight.data, gain=.01)
                # torch.nn.init.normal_(l.bias.data)

                # 다크넷 YOLOv1 가중치 초기화
                scale = torch.sqrt(torch.tensor(2./(l.in_channels * (l.kernel_size[0] ** 2))))
                l.weight.data = 2 * scale * torch.rand(l.weight.size()) - scale

                torch.nn.init.zeros_(l.bias.data)
            elif isinstance(l, nn.Linear):
                # torch.nn.init.kaiming_normal_(l.weight.data)
                # torch.nn.init.xavier_normal_(l.weight.data, gain=.01)
                # torch.nn.init.normal_(l.bias.data)

                # 다크넷 YOLOv1 가중치 초기화
                scale = torch.sqrt(torch.tensor(2./l.in_features))
                l.weight.data = 2 * scale * torch.rand(l.weight.size()) - scale

                torch.nn.init.zeros_(l.bias.data)
        # last_l = self.added_layers[-2]
        # torch.nn.init.kaiming_normal_(last_l.weight.data)
        # torch.nn.init.normal_(last_l.bias.data)

    def normalization_config(self):
        return {key: self.backbone_config[key] for key in ['mean', 'std']}


if __name__ == '__main__':
    from torchsummary import torchsummary

    yolo = YOLOv1(backbone='resnet18')
    fake_inp = torch.rand(1, 3, 448, 448)
    print(yolo(fake_inp).size())
    assert tuple(yolo(torch.rand(1, 3, 448, 448)).size()) == (1, 7, 7, 30)

    torchsummary.summary(yolo, fake_inp)
