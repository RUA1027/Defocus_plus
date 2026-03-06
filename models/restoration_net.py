import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False), nn.InstanceNorm2d(mid_channels, affine=True), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.InstanceNorm2d(out_channels, affine=True), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class SFTLayer(nn.Module):

    def __init__(self, in_channels, cond_channels=36):
        super().__init__()
        hidden_channels = max(32, in_channels // 2)
        self.shared = nn.Sequential(nn.Conv2d(cond_channels, hidden_channels, kernel_size=1, bias=True), nn.LeakyReLU(0.2, inplace=True))
        self.to_gamma = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=True)
        self.to_beta = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_beta.weight)
        if self.to_gamma.bias is not None:
            nn.init.zeros_(self.to_gamma.bias)
        if self.to_beta.bias is not None:
            nn.init.zeros_(self.to_beta.bias)

    def forward(self, x, cond):
        cond = F.interpolate(cond, size=x.shape[2:], mode='bilinear', align_corners=False)
        feat = self.shared(cond)
        gamma = self.to_gamma(feat)
        beta = self.to_beta(feat)
        return x * (1.0 + gamma) + beta

class RestorationNet(nn.Module):

    def __init__(self, n_channels, n_classes, base_filters=32, bilinear=True, use_coords=False, n_coeffs=0):
        super(RestorationNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_coords = use_coords
        self.n_coeffs = n_coeffs
        factor = 2 if bilinear else 1
        input_channels = n_channels + (2 if use_coords else 0)
        self.inc = DoubleConv(input_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        self.down4 = Down(base_filters * 8, base_filters * 16 // factor)
        self.up1 = Up(base_filters * 16, base_filters * 8 // factor, bilinear)
        self.up2 = Up(base_filters * 8, base_filters * 4 // factor, bilinear)
        self.up3 = Up(base_filters * 4, base_filters * 2 // factor, bilinear)
        self.up4 = Up(base_filters * 2, base_filters, bilinear)
        if n_coeffs > 0:
            self.sft1 = SFTLayer(base_filters * 8 // factor, cond_channels=n_coeffs)
            self.sft2 = SFTLayer(base_filters * 4 // factor, cond_channels=n_coeffs)
            self.sft3 = SFTLayer(base_filters * 2 // factor, cond_channels=n_coeffs)
            self.sft4 = SFTLayer(base_filters, cond_channels=n_coeffs)
        self.outc = OutConv(base_filters, n_classes)

    def forward(self, x_input, coeffs_map=None):
        x = x_input
        if self.use_coords:
            (B, C, H, W) = x.shape
            y_coords = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)
            x_coords = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype)
            (grid_y, grid_x) = torch.meshgrid(y_coords, x_coords, indexing='ij')
            grid_y = grid_y.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)
            grid_x = grid_x.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)
            x = torch.cat([x, grid_y, grid_x], dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        use_sft = coeffs_map is not None and self.n_coeffs > 0
        d1 = self.up1(x5, x4)
        if use_sft:
            d1 = self.sft1(d1, coeffs_map)
        d2 = self.up2(d1, x3)
        if use_sft:
            d2 = self.sft2(d2, coeffs_map)
        d3 = self.up3(d2, x2)
        if use_sft:
            d3 = self.sft3(d3, coeffs_map)
        d4 = self.up4(d3, x1)
        if use_sft:
            d4 = self.sft4(d4, coeffs_map)
        logits = self.outc(d4)
        return x_input + logits