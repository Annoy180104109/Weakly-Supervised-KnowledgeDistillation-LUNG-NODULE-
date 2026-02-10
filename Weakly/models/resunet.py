# Weakly/teacher.py
# Memory-friendly ResUNet teacher for 2D binary segmentation.
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
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.act(self.bn(self.conv(x))))


class ResBlock(nn.Module):
    """
    Residual block:
      (Conv-BN-ReLU) x2 + residual 1x1 if channels change.
    """
    def __init__(self, in_ch: int, out_ch: int, drop: float = 0.0):
        super().__init__()
        self.c1 = ConvBNReLU(in_ch, out_ch, drop=drop)
        self.c2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.act = nn.ReLU(inplace=True)

        self.skip = nn.Identity() if in_ch == out_ch else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        h = self.c1(x)
        h = self.c2(h)
        h = self.drop(h)
        return self.act(h + self.skip(x))


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


# ---------------------- ResUNet ----------------------
class ResUNet(nn.Module):
    """
    2D ResUNet (U-Net with residual blocks).
    Depth=4 (down 4 times) like a standard U-Net.

    Encoder: ResBlocks
    Decoder: bilinear upsample + concat + ResBlock
    """
    def __init__(
        self,
        in_chans: int = 1,
        num_classes: int = 1,
        base_ch: int = 32,
        drop: float = 0.0,
    ):
        super().__init__()
        ch = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16]

        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        self.enc0 = ResBlock(in_chans, ch[0], drop=drop)
        self.enc1 = ResBlock(ch[0], ch[1], drop=drop)
        self.enc2 = ResBlock(ch[1], ch[2], drop=drop)
        self.enc3 = ResBlock(ch[2], ch[3], drop=drop)
        self.bott = ResBlock(ch[3], ch[4], drop=drop)

        # Ups
        self.up4 = Up(ch[4], ch[3])
        self.up3 = Up(ch[3], ch[2])
        self.up2 = Up(ch[2], ch[1])
        self.up1 = Up(ch[1], ch[0])

        # Decoder (concat + ResBlock)
        self.dec3 = ResBlock(ch[3] + ch[3], ch[3], drop=drop)
        self.dec2 = ResBlock(ch[2] + ch[2], ch[2], drop=drop)
        self.dec1 = ResBlock(ch[1] + ch[1], ch[1], drop=drop)
        self.dec0 = ResBlock(ch[0] + ch[0], ch[0], drop=drop)

        self.head = nn.Conv2d(ch[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e0 = self.enc0(x)            # H, W
        e1 = self.enc1(self.pool(e0))# H/2
        e2 = self.enc2(self.pool(e1))# H/4
        e3 = self.enc3(self.pool(e2))# H/8
        b  = self.bott(self.pool(e3))# H/16

        # Decoder
        d3 = self.up4(b, e3.shape[-2:])
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up3(d3, e2.shape[-2:])
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up2(d2, e1.shape[-2:])
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        d0 = self.up1(d1, e0.shape[-2:])
        d0 = self.dec0(torch.cat([d0, e0], dim=1))

        return self.head(d0)


def make_teacher_swin_unet(in_ch=1, out_ch=1, img_size=256):
    """
    Backwards-compatible factory name (your notebook expects this).
    Now returns ResUNet.
    img_size is unused but kept for API compatibility.
    """
    _ = img_size
    return ResUNet(in_chans=in_ch, num_classes=out_ch, base_ch=32, drop=0.0)
