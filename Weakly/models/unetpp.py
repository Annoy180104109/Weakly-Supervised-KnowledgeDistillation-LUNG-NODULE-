# Weakly/teacher.py
# Memory-friendly UNet++ (Nested U-Net) teacher for 2D binary segmentation.
# Input : [B, 1, H, W]
# Output: [B, 1, H, W] logits

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------- building blocks ----------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1, drop: float = 0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.act(self.bn(self.conv(x))))


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, drop: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, drop=drop),
            ConvBNReLU(out_ch, out_ch, drop=drop),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """
    Memory-friendly upsample: bilinear + 1x1 conv (instead of transposed conv).
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x, target_hw):
        x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        return self.conv1x1(x)


# ---------------------- UNet++ ----------------------
class UNetPlusPlus(nn.Module):
    """
    UNet++ (Nested U-Net) for binary segmentation.

    Depth=4 (levels 0..4):
      x_{i,j} is node at depth i and stage j (j = nesting level).
    We build nodes:
      x00, x10, x20, x30, x40
      x01, x11, x21, x31
      x02, x12, x22
      x03, x13
      x04

    Deep supervision optional (off by default).
    """
    def __init__(
        self,
        in_chans: int = 1,
        num_classes: int = 1,
        base_ch: int = 32,          # reduce (e.g., 16/24/32) for small GPUs
        drop: float = 0.0,
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.deep_supervision = bool(deep_supervision)

        ch = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16]

        self.pool = nn.MaxPool2d(2, 2)

        # encoder (j=0)
        self.conv00 = DoubleConv(in_chans, ch[0], drop=drop)
        self.conv10 = DoubleConv(ch[0], ch[1], drop=drop)
        self.conv20 = DoubleConv(ch[1], ch[2], drop=drop)
        self.conv30 = DoubleConv(ch[2], ch[3], drop=drop)
        self.conv40 = DoubleConv(ch[3], ch[4], drop=drop)

        # upsamplers (map deeper channels -> shallower channels)
        self.up10 = Up(ch[1], ch[0])
        self.up20 = Up(ch[2], ch[1])
        self.up30 = Up(ch[3], ch[2])
        self.up40 = Up(ch[4], ch[3])

        self.up11 = Up(ch[1], ch[0])
        self.up21 = Up(ch[2], ch[1])
        self.up31 = Up(ch[3], ch[2])

        self.up12 = Up(ch[1], ch[0])
        self.up22 = Up(ch[2], ch[1])

        self.up13 = Up(ch[1], ch[0])

        # decoder nested convs
        # x01 uses x00 + up(x10)
        self.conv01 = DoubleConv(ch[0] + ch[0], ch[0], drop=drop)
        self.conv11 = DoubleConv(ch[1] + ch[1], ch[1], drop=drop)
        self.conv21 = DoubleConv(ch[2] + ch[2], ch[2], drop=drop)
        self.conv31 = DoubleConv(ch[3] + ch[3], ch[3], drop=drop)

        # x02 uses x00 + x01 + up(x11)
        self.conv02 = DoubleConv(ch[0] + ch[0] + ch[0], ch[0], drop=drop)
        self.conv12 = DoubleConv(ch[1] + ch[1] + ch[1], ch[1], drop=drop)
        self.conv22 = DoubleConv(ch[2] + ch[2] + ch[2], ch[2], drop=drop)

        # x03 uses x00 + x01 + x02 + up(x12)
        self.conv03 = DoubleConv(ch[0] * 4, ch[0], drop=drop)
        self.conv13 = DoubleConv(ch[1] * 4, ch[1], drop=drop)

        # x04 uses x00 + x01 + x02 + x03 + up(x13)
        self.conv04 = DoubleConv(ch[0] * 5, ch[0], drop=drop)

        # heads
        if self.deep_supervision:
            self.head1 = nn.Conv2d(ch[0], num_classes, kernel_size=1)
            self.head2 = nn.Conv2d(ch[0], num_classes, kernel_size=1)
            self.head3 = nn.Conv2d(ch[0], num_classes, kernel_size=1)
            self.head4 = nn.Conv2d(ch[0], num_classes, kernel_size=1)
        else:
            self.head = nn.Conv2d(ch[0], num_classes, kernel_size=1)

    def forward(self, x):
        # encoder
        x00 = self.conv00(x)
        x10 = self.conv10(self.pool(x00))
        x20 = self.conv20(self.pool(x10))
        x30 = self.conv30(self.pool(x20))
        x40 = self.conv40(self.pool(x30))

        # decoder (nested)
        x01 = self.conv01(torch.cat([x00, self.up10(x10, x00.shape[-2:])], dim=1))
        x11 = self.conv11(torch.cat([x10, self.up20(x20, x10.shape[-2:])], dim=1))
        x21 = self.conv21(torch.cat([x20, self.up30(x30, x20.shape[-2:])], dim=1))
        x31 = self.conv31(torch.cat([x30, self.up40(x40, x30.shape[-2:])], dim=1))

        x02 = self.conv02(torch.cat([x00, x01, self.up11(x11, x00.shape[-2:])], dim=1))
        x12 = self.conv12(torch.cat([x10, x11, self.up21(x21, x10.shape[-2:])], dim=1))
        x22 = self.conv22(torch.cat([x20, x21, self.up31(x31, x20.shape[-2:])], dim=1))

        x03 = self.conv03(torch.cat([x00, x01, x02, self.up12(x12, x00.shape[-2:])], dim=1))
        x13 = self.conv13(torch.cat([x10, x11, x12, self.up22(x22, x10.shape[-2:])], dim=1))

        x04 = self.conv04(torch.cat([x00, x01, x02, x03, self.up13(x13, x00.shape[-2:])], dim=1))

        if self.deep_supervision:
            # return list of logits at multiple depths (same H,W as input)
            return [
                self.head1(x01),
                self.head2(x02),
                self.head3(x03),
                self.head4(x04),
            ]
        return self.head(x04)


def make_teacher_swin_unet(in_ch=1, out_ch=1, img_size=256):
    """
    Backwards-compatible factory name (your notebook expects this).
    Now returns UNet++.
    img_size is not required by UNet++ but kept for API compatibility.
    """
    _ = img_size
    return UNetPlusPlus(in_chans=in_ch, num_classes=out_ch, base_ch=32, drop=0.0, deep_supervision=False)
