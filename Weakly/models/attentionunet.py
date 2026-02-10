# Weakly/teacher.py
# Attention U-Net teacher for 2D binary segmentation.
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


class DoubleConv(nn.Module):
    """(Conv-BN-ReLU) x2"""
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


class AttentionGate(nn.Module):
    """
    Attention gate from Attention U-Net (Oktay et al.)
    g: gating signal from decoder (coarser)
    x: skip features from encoder (finer)
    returns: gated skip features (x * alpha)
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g and x must be same spatial size
        h = self.act(self.W_g(g) + self.W_x(x))
        alpha = self.psi(h)
        return x * alpha


# ---------------------- Attention U-Net ----------------------
class AttentionUNet(nn.Module):
    """
    2D Attention U-Net.
    Depth=4 (down 4 times) like standard U-Net.
    Encoder: DoubleConv
    Decoder: bilinear upsample + attention-gated skip + DoubleConv
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
        self.enc0 = DoubleConv(in_chans, ch[0], drop=drop)
        self.enc1 = DoubleConv(ch[0], ch[1], drop=drop)
        self.enc2 = DoubleConv(ch[1], ch[2], drop=drop)
        self.enc3 = DoubleConv(ch[2], ch[3], drop=drop)
        self.bott = DoubleConv(ch[3], ch[4], drop=drop)

        # Ups + Attention gates
        self.up4 = Up(ch[4], ch[3])
        self.ag3 = AttentionGate(F_g=ch[3], F_l=ch[3], F_int=ch[2])
        self.dec3 = DoubleConv(ch[3] + ch[3], ch[3], drop=drop)

        self.up3 = Up(ch[3], ch[2])
        self.ag2 = AttentionGate(F_g=ch[2], F_l=ch[2], F_int=ch[1])
        self.dec2 = DoubleConv(ch[2] + ch[2], ch[2], drop=drop)

        self.up2 = Up(ch[2], ch[1])
        self.ag1 = AttentionGate(F_g=ch[1], F_l=ch[1], F_int=ch[0])
        self.dec1 = DoubleConv(ch[1] + ch[1], ch[1], drop=drop)

        self.up1 = Up(ch[1], ch[0])
        self.ag0 = AttentionGate(F_g=ch[0], F_l=ch[0], F_int=max(ch[0] // 2, 8))
        self.dec0 = DoubleConv(ch[0] + ch[0], ch[0], drop=drop)

        self.head = nn.Conv2d(ch[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e0 = self.enc0(x)             # H, W
        e1 = self.enc1(self.pool(e0)) # H/2
        e2 = self.enc2(self.pool(e1)) # H/4
        e3 = self.enc3(self.pool(e2)) # H/8
        b  = self.bott(self.pool(e3)) # H/16

        # Decoder + attention gated skips
        d3 = self.up4(b, e3.shape[-2:])
        s3 = self.ag3(g=d3, x=e3)
        d3 = self.dec3(torch.cat([d3, s3], dim=1))

        d2 = self.up3(d3, e2.shape[-2:])
        s2 = self.ag2(g=d2, x=e2)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))

        d1 = self.up2(d2, e1.shape[-2:])
        s1 = self.ag1(g=d1, x=e1)
        d1 = self.dec1(torch.cat([d1, s1], dim=1))

        d0 = self.up1(d1, e0.shape[-2:])
        s0 = self.ag0(g=d0, x=e0)
        d0 = self.dec0(torch.cat([d0, s0], dim=1))

        return self.head(d0)


def make_teacher_swin_unet(in_ch=1, out_ch=1, img_size=256):
    """
    Backwards-compatible factory name (your notebook expects this).
    Now returns AttentionUNet.
    img_size is unused but kept for API compatibility.
    """
    _ = img_size
    return AttentionUNet(in_chans=in_ch, num_classes=out_ch, base_ch=32, drop=0.0)
