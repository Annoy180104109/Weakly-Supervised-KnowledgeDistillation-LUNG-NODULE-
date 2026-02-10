# Weakly/student_dwunet_aspp.py
# Student: lightweight depthwise-separable U-Net + ASPP at bottleneck.
# Input:  [B, 1, H, W]
# Output: [B, 1, H, W] logits
#
# Feature distillation support:
#   forward(x, return_feats=True) -> (logits, (d1,d2,d3,d4,bt))
# where d1..d4 are encoder feature maps (NCHW) and bt is the *post-ASPP* bottleneck.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DWSeparableBlock(nn.Module):
    """
    Depthwise-separable residual block at fixed channel width C:
    1x1 reduce -> DW 3x3 -> 1x1 expand, BN+ReLU with residual.
    """
    def __init__(self, C: int, r: int = 4):
        super().__init__()
        Cmid = max(8, C // r)
        self.reduce = nn.Conv2d(C, Cmid, 1, bias=False)
        self.dw     = nn.Conv2d(Cmid, Cmid, 3, padding=1, groups=Cmid, bias=False)
        self.expand = nn.Conv2d(Cmid, C, 1, bias=False)
        self.bn     = nn.BatchNorm2d(C)
        self.act    = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.reduce(x)
        h = self.dw(h)
        h = self.expand(h)
        return self.act(self.bn(h + x))


class DWDoubleConvSame(nn.Module):
    """Two DWSeparableBlocks at the same channel width C (encoder inner blocks)."""
    def __init__(self, C: int, r: int = 4):
        super().__init__()
        self.block = nn.Sequential(
            DWSeparableBlock(C, r=r),
            DWSeparableBlock(C, r=r),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DWDoubleConvIO(nn.Module):
    """
    Depthwise-separable 'DoubleConv' with in/out mapping:
    pre 1x1: in_c -> out_c (only if needed)
    then two DWSeparableBlocks at width out_c.
    """
    def __init__(self, in_c: int, out_c: int, r: int = 4):
        super().__init__()
        self.pre = nn.Identity() if in_c == out_c else nn.Conv2d(in_c, out_c, 1, bias=False)
        self.block = nn.Sequential(
            DWSeparableBlock(out_c, r=r),
            DWSeparableBlock(out_c, r=r),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        return self.block(x)


class ASPP(nn.Module):
    """
    ASPP for small bottleneck maps (e.g., 16x16 when input is 256x256 with 4 pools).
    Keeps channels the same: C -> C.

    Recommended rates for 256x256 input here: (1, 2, 4, 6)
    """
    def __init__(self, C: int, rates=(1, 2, 4, 6)):
        super().__init__()
        assert len(rates) == 4, "rates must be a 4-tuple like (1,2,4,6)"

        r0, r1, r2, r3 = rates

        self.b0 = nn.Sequential(
            nn.Conv2d(C, C, 1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )

        def atrous3x3(d: int):
            return nn.Sequential(
                nn.Conv2d(C, C, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True),
            )

        self.b1 = atrous3x3(r1)
        self.b2 = atrous3x3(r2)
        self.b3 = atrous3x3(r3)

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C, C, 1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )

        self.proj = nn.Sequential(
            nn.Conv2d(C * 5, C, 1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        p = self.pool(x)
        p = F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)
        y = torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x), p], dim=1)
        return self.proj(y)


class DWUNet(nn.Module):
    """
    Student: lightweight DW U-Net with ASPP at bottleneck.
    Stem/head remain standard. Encoder inner blocks use DWDoubleConvSame.
    Decoder uses DWDoubleConvIO to map (concat) -> target channels.

    If return_feats=True:
      returns (logits, (d1, d2, d3, d4, bt))
    where bt is the *post-ASPP* bottleneck.
    """
    def __init__(
        self,
        in_ch: int = 1,
        base: int = 32,
        out_ch: int = 1,
        r: int = 4,
        aspp_rates=(1, 2, 4, 6),
    ):
        super().__init__()
        b = base

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, b, 3, padding=1, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
        )

        # Encoder
        self.d1 = DWDoubleConvSame(b, r=r)
        self.p1 = nn.MaxPool2d(2)

        self.enc2_proj = nn.Conv2d(b,   b * 2, 1, bias=False)
        self.d2 = DWDoubleConvSame(b * 2, r=r)
        self.p2 = nn.MaxPool2d(2)

        self.enc3_proj = nn.Conv2d(b * 2, b * 4, 1, bias=False)
        self.d3 = DWDoubleConvSame(b * 4, r=r)
        self.p3 = nn.MaxPool2d(2)

        self.enc4_proj = nn.Conv2d(b * 4, b * 8, 1, bias=False)
        self.d4 = DWDoubleConvSame(b * 8, r=r)
        self.p4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bt_proj = nn.Conv2d(b * 8, b * 16, 1, bias=False)
        self.bt      = DWDoubleConvSame(b * 16, r=r)

        # ASPP inserted here (bt -> aspp -> decoder)
        self.aspp = ASPP(b * 16, rates=aspp_rates)

        # Decoder
        self.u4 = nn.ConvTranspose2d(b * 16, b * 8, 2, 2)
        self.c4 = DWDoubleConvIO(in_c=b * 8 + b * 8, out_c=b * 8, r=r)

        self.u3 = nn.ConvTranspose2d(b * 8,  b * 4, 2, 2)
        self.c3 = DWDoubleConvIO(in_c=b * 4 + b * 4, out_c=b * 4, r=r)

        self.u2 = nn.ConvTranspose2d(b * 4,  b * 2, 2, 2)
        self.c2 = DWDoubleConvIO(in_c=b * 2 + b * 2, out_c=b * 2, r=r)

        self.u1 = nn.ConvTranspose2d(b * 2,  b,      2, 2)
        self.c1 = DWDoubleConvIO(in_c=b + b, out_c=b, r=r)

        self.out = nn.Conv2d(b, out_ch, 1)

    def forward(self, x: torch.Tensor, return_feats: bool = False):
        d1s = self.stem(x)
        d1  = self.d1(d1s)
        p1  = self.p1(d1)

        e2  = self.enc2_proj(p1)
        d2  = self.d2(e2)
        p2  = self.p2(d2)

        e3  = self.enc3_proj(p2)
        d3  = self.d3(e3)
        p3  = self.p3(d3)

        e4  = self.enc4_proj(p3)
        d4  = self.d4(e4)
        p4  = self.p4(d4)

        bt_i = self.bt_proj(p4)
        bt   = self.bt(bt_i)
        bt   = self.aspp(bt)  # <-- ASPP at bottleneck (for 256x256 input, bt is 16x16)

        y = self.u4(bt)
        y = self.c4(torch.cat([y, d4], 1))

        y = self.u3(y)
        y = self.c3(torch.cat([y, d3], 1))

        y = self.u2(y)
        y = self.c2(torch.cat([y, d2], 1))

        y = self.u1(y)
        y = self.c1(torch.cat([y, d1], 1))

        logits = self.out(y)

        if return_feats:
            return logits, (d1, d2, d3, d4, bt)
        return logits


def make_student_dwunet(
    in_ch: int = 1,
    out_ch: int = 1,
    base: int = 32,
    r: int = 4,
    aspp_rates=(1, 2, 4, 6),
) -> DWUNet:
    return DWUNet(in_ch=in_ch, out_ch=out_ch, base=base, r=r, aspp_rates=aspp_rates)
