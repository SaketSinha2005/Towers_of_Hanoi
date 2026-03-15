"""
models/unet.py — U-Net with Attention Gates
Supports standard U-Net, Attention U-Net, and U-Net++
Input:  [B, 4, H, W]  (4 MRI modalities)
Output: [B, 4, H, W]  (4 class logits: background, necrotic, edema, enhancing)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─── Building Blocks ──────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    """(Conv → BN → ReLU) × 2"""
    def __init__(self, in_ch: int, out_ch: int, mid_ch: int = None, dropout: float = 0.0):
        super().__init__()
        mid_ch = mid_ch or out_ch
        layers = [
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers += [
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """MaxPool → DoubleConv (encoder step)"""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, dropout=dropout)
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Upsample → concat skip → DoubleConv (decoder step)"""
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = False, dropout: float = 0.0):
        super().__init__()
        if bilinear:
            self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2, dropout=dropout)
        else:
            self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch, dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad if sizes differ
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# ─── Attention Gate ───────────────────────────────────────────────────────────
class AttentionGate(nn.Module):
    """
    Attention gate for skip connections.
    Lets the model focus on tumor-relevant spatial regions.
    """
    def __init__(self, g_ch: int, x_ch: int, inter_ch: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_ch, inter_ch, 1, bias=True),
            nn.BatchNorm2d(inter_ch)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_ch, inter_ch, 1, bias=True),
            nn.BatchNorm2d(inter_ch)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Upsample g to match x resolution
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=True)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# ─── U-Net ────────────────────────────────────────────────────────────────────
class UNet(nn.Module):
    """
    Standard U-Net for brain tumor segmentation.
    in_channels  = 4  (t1, t1ce, t2, flair)
    out_channels = 4  (bg, necrotic, edema, enhancing)

    Named encoder layers for Grad-CAM targeting:
      self.encoder0 … self.encoder4
    """
    def __init__(self, in_channels: int = config.IN_CHANNELS,
                 out_channels: int = config.OUT_CHANNELS,
                 features: list = config.FEATURES,
                 bilinear: bool = config.BILINEAR,
                 dropout: float = config.DROPOUT):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoder0 = DoubleConv(in_channels, features[0], dropout=dropout)
        self.encoder1 = Down(features[0], features[1], dropout=dropout)
        self.encoder2 = Down(features[1], features[2], dropout=dropout)
        self.encoder3 = Down(features[2], features[3], dropout=dropout)
        # Bottleneck
        self.encoder4 = Down(features[3], features[4] if not bilinear
                             else features[4] // 2, dropout=dropout)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.decoder4 = Up(features[4], features[3] if not bilinear
                           else features[3], bilinear, dropout=dropout)
        self.decoder3 = Up(features[3], features[2], bilinear, dropout=dropout)
        self.decoder2 = Up(features[2], features[1], bilinear, dropout=dropout)
        self.decoder1 = Up(features[1], features[0], bilinear, dropout=dropout)

        # ── Output ───────────────────────────────────────────────────────────
        self.out_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        # Encoder
        e0 = self.encoder0(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)      # bottleneck

        # Decoder + skip connections
        d4 = self.decoder4(e4, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2, e0)

        return self.out_conv(d1)    # [B, C, H, W]


# ─── Attention U-Net ──────────────────────────────────────────────────────────
class AttentionUNet(nn.Module):
    """U-Net with Attention Gates on skip connections."""
    def __init__(self, in_channels: int = config.IN_CHANNELS,
                 out_channels: int = config.OUT_CHANNELS,
                 features: list = config.FEATURES,
                 dropout: float = config.DROPOUT):
        super().__init__()
        f = features
        # Encoder
        self.encoder0 = DoubleConv(in_channels, f[0])
        self.encoder1 = Down(f[0], f[1])
        self.encoder2 = Down(f[1], f[2])
        self.encoder3 = Down(f[2], f[3])
        self.encoder4 = Down(f[3], f[4])   # bottleneck

        # Attention gates
        self.attn4 = AttentionGate(f[4], f[3], f[3] // 2)
        self.attn3 = AttentionGate(f[3], f[2], f[2] // 2)
        self.attn2 = AttentionGate(f[2], f[1], f[1] // 2)
        self.attn1 = AttentionGate(f[1], f[0], f[0] // 2)

        # Decoder
        self.decoder4 = Up(f[4], f[3], dropout=dropout)
        self.decoder3 = Up(f[3], f[2], dropout=dropout)
        self.decoder2 = Up(f[2], f[1], dropout=dropout)
        self.decoder1 = Up(f[1], f[0], dropout=dropout)

        self.out_conv = nn.Conv2d(f[0], out_channels, 1)

    def forward(self, x):
        e0 = self.encoder0(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Attention-gated skip connections
        a3 = self.attn4(e4, e3)
        d4 = self.decoder4(e4, a3)
        a2 = self.attn3(d4, e2)
        d3 = self.decoder3(d4, a2)
        a1 = self.attn2(d3, e1)
        d2 = self.decoder2(d3, a1)
        a0 = self.attn1(d2, e0)
        d1 = self.decoder1(d2, a0)

        return self.out_conv(d1)


# ─── Factory ──────────────────────────────────────────────────────────────────
def build_model(name: str = config.MODEL_NAME) -> nn.Module:
    name = name.lower()
    if name == "unet":
        model = UNet()
    elif name in ("attention_unet", "attention-unet", "attunet"):
        model = AttentionUNet()
    else:
        raise ValueError(f"Unknown model: {name}. Choose 'unet' or 'attention_unet'.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {name} | Parameters: {n_params:,}")
    return model


if __name__ == "__main__":
    model = build_model("attention_unet")
    x = torch.randn(2, 4, 128, 128)
    y = model(x)
    print(f"Input: {x.shape} → Output: {y.shape}")