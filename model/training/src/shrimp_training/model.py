# SPDX-License-Identifier: AGPL-3.0-or-later
"""YOLO detector built from scratch in PyTorch -- no Ultralytics dependency.

Backbone (Darknet-style) -> Neck (FPN) -> Head (3 detection scales).

Output at each scale: ``(B, num_anchors, H, W, 5 + num_classes)`` where
``5 = [tx, ty, tw, th, objectness]``. The three scales run at strides 8, 16 and 32,
matching the anchor-count math already used throughout this project
(``expected_anchor_count`` in the runtime's ``detection/decode.py``).

This module only defines the network. Anchor box sizes, loss, training loop,
evaluation and ONNX export live in their own modules (``anchors.py``, ``loss.py``,
``adapter.py``) so this stays a pure, side-effect-free ``nn.Module`` definition.
"""

from __future__ import annotations

import torch
from torch import nn

#: Detection heads run at these strides, in scale order (finest first). This order is
#: load-bearing: the backend registry pins anchors 0-2 to stride 8, 3-5 to stride 16
#: and 6-8 to stride 32.
STRIDES: tuple[int, ...] = (8, 16, 32)

# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------


class ConvBNSiLU(nn.Module):
    """Conv -> BatchNorm -> SiLU. The fundamental unit of the whole network."""

    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.act(self.bn(self.conv(x)))
        return result


class ResidualBlock(nn.Module):
    """Squeeze channels, expand back, add the input. Lets gradients skip layers."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBNSiLU(c, c // 2, k=1),
            ConvBNSiLU(c // 2, c, k=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = x + self.block(x)
        return result


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast.

    Pools at several receptive-field sizes so the model sees multiple scales at once.
    """

    def __init__(self, c_in: int, c_out: int, k: int = 5) -> None:
        super().__init__()
        c_hidden = c_in // 2
        self.cv1 = ConvBNSiLU(c_in, c_hidden, k=1)
        self.cv2 = ConvBNSiLU(c_hidden * 4, c_out, k=1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        result: torch.Tensor = self.cv2(torch.cat([x, y1, y2, y3], dim=1))
        return result


def conv_set(c_in: int, c_out: int) -> nn.Sequential:
    """The 1x1 / 3x3 alternating stack used before every detection head."""
    return nn.Sequential(
        ConvBNSiLU(c_in, c_out, k=1),
        ConvBNSiLU(c_out, c_out * 2, k=3),
        ConvBNSiLU(c_out * 2, c_out, k=1),
    )


# --------------------------------------------------------------------------
# Full model
# --------------------------------------------------------------------------


class YOLO(nn.Module):
    def __init__(self, num_classes: int = 2, num_anchors: int = 3) -> None:
        super().__init__()
        self.nc = num_classes
        self.na = num_anchors
        self.no = num_classes + 5  # outputs per anchor
        self.nl = 3  # number of detection layers

        # ---------------- Backbone ----------------
        self.stem = ConvBNSiLU(3, 32, k=3, s=1)

        self.down1 = ConvBNSiLU(32, 64, k=3, s=2)
        self.res1 = nn.Sequential(*[ResidualBlock(64) for _ in range(1)])

        self.down2 = ConvBNSiLU(64, 128, k=3, s=2)
        self.res2 = nn.Sequential(*[ResidualBlock(128) for _ in range(2)])

        self.down3 = ConvBNSiLU(128, 256, k=3, s=2)  # -> P3 (stride 8)
        self.res3 = nn.Sequential(*[ResidualBlock(256) for _ in range(3)])

        self.down4 = ConvBNSiLU(256, 512, k=3, s=2)  # -> P4 (stride 16)
        self.res4 = nn.Sequential(*[ResidualBlock(512) for _ in range(3)])

        self.down5 = ConvBNSiLU(512, 1024, k=3, s=2)  # -> P5 (stride 32)
        self.res5 = nn.Sequential(*[ResidualBlock(1024) for _ in range(2)])

        self.sppf = SPPF(1024, 1024)

        # ---------------- Neck (top-down FPN) ----------------
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        self.set5 = conv_set(1024, 512)
        self.reduce5 = ConvBNSiLU(512, 256, k=1)

        self.set4 = conv_set(256 + 512, 256)  # concat with P4
        self.reduce4 = ConvBNSiLU(256, 128, k=1)

        self.set3 = conv_set(128 + 256, 128)  # concat with P3

        # ---------------- Heads ----------------
        self.head_p5 = nn.Sequential(
            ConvBNSiLU(512, 1024, k=3),
            nn.Conv2d(1024, self.na * self.no, kernel_size=1),
        )
        self.head_p4 = nn.Sequential(
            ConvBNSiLU(256, 512, k=3),
            nn.Conv2d(512, self.na * self.no, kernel_size=1),
        )
        self.head_p3 = nn.Sequential(
            ConvBNSiLU(128, 256, k=3),
            nn.Conv2d(256, self.na * self.no, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        # Bias init for the objectness channel: start by predicting "mostly empty".
        # Without this, early training is dominated by false positives.
        for head in (self.head_p3, self.head_p4, self.head_p5):
            final = head[-1]
            assert isinstance(final, nn.Conv2d)
            assert final.bias is not None
            b = final.bias.view(self.na, -1)
            with torch.no_grad():
                b[:, 4] += -4.6  # sigmoid(-4.6) ~ 0.01
                b[:, 5:] += -4.6
            final.bias = nn.Parameter(b.view(-1), requires_grad=True)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        """(B, na*no, H, W) -> (B, na, H, W, no)"""
        b, _, h, w = x.shape
        x = x.view(b, self.na, self.no, h, w)
        return x.permute(0, 1, 3, 4, 2).contiguous()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # Backbone
        x = self.stem(x)
        x = self.res1(self.down1(x))
        x = self.res2(self.down2(x))
        p3 = self.res3(self.down3(x))  # stride 8
        p4 = self.res4(self.down4(p3))  # stride 16
        p5 = self.res5(self.down5(p4))  # stride 32
        p5 = self.sppf(p5)

        # Neck: deep features flow back up to the high-resolution maps
        t5 = self.set5(p5)
        out5 = self.head_p5(t5)

        u4 = self.up(self.reduce5(t5))
        t4 = self.set4(torch.cat([u4, p4], dim=1))
        out4 = self.head_p4(t4)

        u3 = self.up(self.reduce4(t4))
        t3 = self.set3(torch.cat([u3, p3], dim=1))
        out3 = self.head_p3(t3)

        # Ordered small-stride first: [stride 8, stride 16, stride 32]
        return [self._reshape(out3), self._reshape(out4), self._reshape(out5)]
