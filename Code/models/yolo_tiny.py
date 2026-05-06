import torch
import torch.nn as nn
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=None, activation=nn.LeakyReLU(0.1)):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = activation

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, channels, num_blocks):
        super().__init__()
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                ConvBlock(channels, channels // 2, 1),
                ConvBlock(channels // 2, channels, 3),
            )
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x) + x
        return x


class YOLOv3Tiny(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes

        # Backbone: 7 Conv layers + 6 MaxPool
        self.backbone = nn.Sequential(
            ConvBlock(3, 16, 3, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(16, 32, 3, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(32, 64, 3, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128, 3, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(128, 256, 3, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(256, 512, 3, 1),
            nn.MaxPool2d(2, 1),
            ConvBlock(512, 1024, 3, 1),
        )

        # Neck / detection heads
        self.conv13 = nn.Sequential(
            ConvBlock(1024, 256, 1),
            ConvBlock(256, 512, 3),
        )
        self.out13 = nn.Conv2d(512, 3 * (5 + num_classes), 1)

        self.upsample = nn.Sequential(
            ConvBlock(256, 128, 1),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.conv26 = nn.Sequential(
            ConvBlock(128 + 256, 256, 3),
        )
        self.out26 = nn.Conv2d(256, 3 * (5 + num_classes), 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Route the 26x26 feature map before the last maxpool
        route_26 = self.backbone[:18](x)   # after ConvBlock(256, 512, 3) -> 26x26
        x = self.backbone[18:](route_26)   # MaxPool + ConvBlock(512, 1024) -> 13x13

        # 13x13 scale
        x13 = self.conv13(x)
        out13 = self.out13(x13)

        # 26x26 scale
        x_up = self.upsample(x13)
        x_cat = torch.cat([x_up, route_26], dim=1)
        x26 = self.conv26(x_cat)
        out26 = self.out26(x26)

        return out13, out26


def decode_outputs(output, anchors, num_classes, stride):
    batch_size, _, grid_h, grid_w = output.shape
    num_anchors = len(anchors)
    output = output.view(batch_size, num_anchors, 5 + num_classes, grid_h, grid_w)
    output = output.permute(0, 1, 3, 4, 2).contiguous()

    x = torch.sigmoid(output[..., 0])
    y = torch.sigmoid(output[..., 1])
    w = output[..., 2]
    h = output[..., 3]
    conf = torch.sigmoid(output[..., 4])
    cls_probs = torch.sigmoid(output[..., 5:])

    grid_x = torch.arange(grid_w, device=output.device).view(1, 1, 1, grid_w).float()
    grid_y = torch.arange(grid_h, device=output.device).view(1, 1, grid_h, 1).float()

    bx = (x + grid_x) * stride
    by = (y + grid_y) * stride

    anchor_w = torch.tensor([a[0] for a in anchors], device=output.device).view(1, num_anchors, 1, 1)
    anchor_h = torch.tensor([a[1] for a in anchors], device=output.device).view(1, num_anchors, 1, 1)
    bw = torch.exp(w) * anchor_w
    bh = torch.exp(h) * anchor_h

    return bx, by, bw, bh, conf, cls_probs
