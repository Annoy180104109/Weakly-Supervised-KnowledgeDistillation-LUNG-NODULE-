# Weakly/teacher.py
# SwinUNet teacher for 2D binary segmentation.
# Input : [B, 1, H, W]
# Output: [B, 1, H, W] logits
#
# Feature distillation support:
#   forward(x, return_feats=True) -> (logits, (f0,f1,f2,f3))
# where f0..f3 are Swin encoder features (typically strides 4/8/16/32).

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------- small decoder blocks ----------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1, drop: float = 0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.bn(self.conv(x))))


class DecBlock(nn.Module):
    """Upsample -> concat skip -> 2x ConvBNReLU"""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, drop: float = 0.0):
        super().__init__()
        self.c1 = ConvBNReLU(in_ch + skip_ch, out_ch, drop=drop)
        self.c2 = ConvBNReLU(out_ch, out_ch, drop=drop)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # x: NCHW, skip: NCHW
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.c1(x)
        x = self.c2(x)
        return x


# ---------------------- Swin-UNet ----------------------
class SwinUNet(nn.Module):
    """
    Swin Transformer encoder (timm features_only) + UNet decoder.
    Produces logits at original resolution.

    If return_feats=True:
      returns (logits, (f0, f1, f2, f3)) where f0..f3 are encoder feature maps in NCHW.
    """
    def __init__(
        self,
        in_chans: int = 1,
        out_chans: int = 1,
        img_size: int = 256,
        backbone: str = "swin_tiny_patch4_window7_224",
        drop: float = 0.0,
        pretrained: bool = True,
    ):
        super().__init__()

        # timm backbone
        try:
            import timm
        except Exception as e:
            raise RuntimeError(
                "This SwinUNet implementation requires `timm`.\n"
                "Install it with: pip install timm\n"
            ) from e

        # features_only returns a list of feature maps at multiple scales.
        # For Swin: typically 4 stages with strides: 4, 8, 16, 32 (relative to input).
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=True,
            img_size=img_size,
            out_indices=(0, 1, 2, 3),
        )

        # Encoder channel info (used in forward for NHWC->NCHW detection)
        self.enc_ch = self.encoder.feature_info.channels()  # e.g. [96, 192, 384, 768]

        # Decoder channels
        d3 = self.enc_ch[2]
        d2 = self.enc_ch[1]
        d1 = self.enc_ch[0]
        d0 = max(8, self.enc_ch[0] // 2)  # keep >= 8 channels

        # Bridge
        self.bridge = nn.Sequential(
            ConvBNReLU(self.enc_ch[3], self.enc_ch[3], drop=drop),
            ConvBNReLU(self.enc_ch[3], self.enc_ch[3], drop=drop),
        )

        # UNet-like decoder: deepest -> stage2 -> stage1 -> stage0
        self.dec3 = DecBlock(in_ch=self.enc_ch[3], skip_ch=self.enc_ch[2], out_ch=d3, drop=drop)
        self.dec2 = DecBlock(in_ch=d3,           skip_ch=self.enc_ch[1], out_ch=d2, drop=drop)
        self.dec1 = DecBlock(in_ch=d2,           skip_ch=self.enc_ch[0], out_ch=d1, drop=drop)

        # Final head (at H/4 resolution)
        self.final = nn.Sequential(
            ConvBNReLU(d1, d0, drop=drop),
            nn.Conv2d(d0, out_chans, kernel_size=1),
        )

    @staticmethod
    def _maybe_nhwc_to_nchw(f: torch.Tensor, expected_c: int) -> torch.Tensor:
        """
        timm Swin sometimes outputs features in NHWC (B,H,W,C).
        Convert to NCHW (B,C,H,W) if needed.
        """
        if f.dim() == 4:
            # If already NCHW: f.shape[1] == expected_c
            if f.shape[1] != expected_c and f.shape[-1] == expected_c:
                f = f.permute(0, 3, 1, 2).contiguous()
        return f

    def forward(self, x: torch.Tensor, return_feats: bool = False):
        feats = self.encoder(x)  # list: [f0, f1, f2, f3]
        f0, f1, f2, f3 = feats

        # Convert encoder features to NCHW if timm returned NHWC
        f0 = self._maybe_nhwc_to_nchw(f0, self.enc_ch[0])
        f1 = self._maybe_nhwc_to_nchw(f1, self.enc_ch[1])
        f2 = self._maybe_nhwc_to_nchw(f2, self.enc_ch[2])
        f3 = self._maybe_nhwc_to_nchw(f3, self.enc_ch[3])

        # Decoder
        y = self.bridge(f3)
        y = self.dec3(y, f2)
        y = self.dec2(y, f1)
        y = self.dec1(y, f0)

        logits = self.final(y)

        # f0 is typically H/4, so upscale back to H, W
        logits = F.interpolate(logits, scale_factor=4, mode="bilinear", align_corners=False)

        if return_feats:
            return logits, (f0, f1, f2, f3)
        return logits


def make_teacher_swin_unet(in_ch: int = 1, out_ch: int = 1, img_size: int = 256) -> SwinUNet:
    """
    Backwards-compatible factory name (your notebook expects this).
    Returns SwinUNet.
    """
    return SwinUNet(
        in_chans=in_ch,
        out_chans=out_ch,
        img_size=img_size,
        backbone="swin_tiny_patch4_window7_224",
        drop=0.0,
        pretrained=True,
    )
