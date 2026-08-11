import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

class IdentityAct(nn.Module):
    """x — linear activation, HE depth cost: 0"""
    def forward(self, x):
        return x


class LinearAct(nn.Module):
    """ax — learnable scalar, HE depth cost: 0"""
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.a * x


class SquaredAct(nn.Module):
    """x² — simplest nonlinear polynomial, HE depth cost: 1"""
    def forward(self, x):
        return x * x


class PolyAct(nn.Module):
    """ax² + bx + c — learnable polynomial, HE depth cost: 1
    Inspired by ULD-Net (Xie et al., ICLR 2026).
    Initialization from ULD-Net: c0=0.5, c1=1, c2=0.1
    """
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.1))   # c2
        self.b = nn.Parameter(torch.tensor(1.0))   # c1
        self.c = nn.Parameter(torch.tensor(0.5))   # c0

    def forward(self, x):
        return self.a * x * x + self.b * x + self.c

class PolyNorm(nn.Module):
    """
    PolyNorm stabile: usa 1/sqrt(var + eps) approssimato con clamp aggressivo.
    In inferenza usa running stats fisse -> compatibile HE.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta  = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var',  torch.ones(num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=[0, 2, 3])
            var  = x.var(dim=[0, 2, 3], unbiased=False)
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var
        else:
            mean = self.running_mean
            var  = self.running_var

        mean = mean.view(1, -1, 1, 1)
        var  = var.view(1, -1, 1, 1)

        # Normalizza con var clampata — stabile sempre
        var_clamped = var.clamp(min=self.eps)
        x_norm = (x - mean) / torch.sqrt(var_clamped)

        # Approssimazione polinomiale di 1/sqrt in inferenza
        # In training usiamo sqrt vera per stabilità
        # In inferenza: 1/sqrt(v) ≈ costante precalcolata * (1 - 0.5*(v-1))
        # per v vicino a 1 (dopo normalizzazione running)

        gamma = self.gamma.view(1, -1, 1, 1)
        beta  = self.beta.view(1, -1, 1, 1)
        return gamma * x_norm + beta
    

ACTIVATIONS = {
    'identity': IdentityAct,
    'linear':   LinearAct,
    'squared':  SquaredAct,
    'poly':     PolyAct,
}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def get_norm(norm_type: str, num_features: int):
    if norm_type == 'none':
        return nn.Identity()
    elif norm_type == 'batch':
        return nn.BatchNorm2d(num_features, affine=True)
    elif norm_type == 'poly':
        return PolyNorm(num_features)
    elif norm_type == 'instance':
        return nn.InstanceNorm2d(num_features, affine=True, track_running_stats=True)
    else:
        raise ValueError(f'Unknown norm_type: {norm_type}')


# ---------------------------------------------------------------------------
# Basic building block
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv3x3 → Norm → Act, repeated twice per stage (as in nnU-Net)."""
    def __init__(self, in_ch, out_ch, stride=1,
                 norm_type='none', act_type='poly'):
        super().__init__()
        Act = ACTIVATIONS[act_type]
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=True),
            get_norm(norm_type, out_ch),
            Act(),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=True),
            get_norm(norm_type, out_ch),
            Act(),
        )

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------------------------------
# HE-friendly U-Net
# ---------------------------------------------------------------------------

class HEFriendlyUNet(nn.Module):
    """
    2D U-Net with the same structure as nnU-Net on ACDC:
      - 6 encoder stages, 5 decoder stages
      - Stride convolution instead of MaxPool (HE-compatible)
      - Configurable activation and normalization

    Args:
        in_channels:  number of input channels (1 for cine MRI)
        num_classes:  number of output classes (4 for ACDC)
        act_type:     'identity' | 'linear' | 'squared' | 'poly'
        norm_type:    'none' | 'batch' | 'instance'
    """

    def __init__(self, in_channels=1, num_classes=4,
                 act_type='poly', norm_type='none'):
        super().__init__()

        self.act_type  = act_type
        self.norm_type = norm_type

        filters = [32, 64, 128, 256, 512, 512]

        # Encoder
        self.enc0 = ConvBlock(in_channels, filters[0], stride=1,
                              norm_type=norm_type, act_type=act_type)
        self.enc1 = ConvBlock(filters[0], filters[1], stride=2,
                              norm_type=norm_type, act_type=act_type)
        self.enc2 = ConvBlock(filters[1], filters[2], stride=2,
                              norm_type=norm_type, act_type=act_type)
        self.enc3 = ConvBlock(filters[2], filters[3], stride=2,
                              norm_type=norm_type, act_type=act_type)
        self.enc4 = ConvBlock(filters[3], filters[4], stride=2,
                              norm_type=norm_type, act_type=act_type)
        self.enc5 = ConvBlock(filters[4], filters[5], stride=2,
                              norm_type=norm_type, act_type=act_type)

        # Decoder
        Act = ACTIVATIONS[act_type]

        self.up4 = nn.ConvTranspose2d(filters[5], filters[4], 2, stride=2)
        self.dec4 = ConvBlock(filters[4] + filters[4], filters[4],
                              norm_type=norm_type, act_type=act_type)

        self.up3 = nn.ConvTranspose2d(filters[4], filters[3], 2, stride=2)
        self.dec3 = ConvBlock(filters[3] + filters[3], filters[3],
                              norm_type=norm_type, act_type=act_type)

        self.up2 = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)
        self.dec2 = ConvBlock(filters[2] + filters[2], filters[2],
                              norm_type=norm_type, act_type=act_type)

        self.up1 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.dec1 = ConvBlock(filters[1] + filters[1], filters[1],
                              norm_type=norm_type, act_type=act_type)

        self.up0 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.dec0 = ConvBlock(filters[0] + filters[0], filters[0],
                              norm_type=norm_type, act_type=act_type)

        # Output
        self.out_conv = nn.Conv2d(filters[0], num_classes, 1)

    def forward(self, x):
        # Encoder
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(e5), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), e0], dim=1))

        return self.out_conv(d0)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    for act in ['identity', 'linear', 'squared', 'poly']:
        for norm in ['none', 'batch']:
            model = HEFriendlyUNet(act_type=act, norm_type=norm)
            x = torch.randn(1, 1, 256, 224)
            y = model(x)
            n_params = sum(p.numel() for p in model.parameters())
            print(f'act={act:10s} norm={norm:10s} '
                  f'output={tuple(y.shape)}  params={n_params:,}')