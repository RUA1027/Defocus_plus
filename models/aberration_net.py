import torch
import torch.nn as nn
import numpy as np

class FourierFeatureEncoding(nn.Module):

    def __init__(self, input_dim=2, mapping_size=64, scale=10):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.scale = scale
        self.register_buffer('B', torch.randn((input_dim, mapping_size)) * scale)

    def forward(self, x):
        xp = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(xp), torch.cos(xp)], dim=-1)

class AberrationNet(nn.Module):

    def __init__(self, num_coeffs, hidden_dim, a_max, use_fourier=True, fourier_scale=5, output_raw=False):
        super().__init__()
        self.num_coeffs = num_coeffs
        self.a_max = a_max
        self.use_fourier = use_fourier
        self.output_raw = output_raw
        if use_fourier:
            self.encoding = FourierFeatureEncoding(input_dim=2, mapping_size=hidden_dim // 2, scale=fourier_scale)
            in_dim = hidden_dim
        else:
            in_dim = 2
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LeakyReLU(0.2), nn.Linear(hidden_dim, hidden_dim * 2), nn.LeakyReLU(0.2), nn.Linear(hidden_dim * 2, hidden_dim), nn.LeakyReLU(0.2), nn.Linear(hidden_dim, num_coeffs))
        nn.init.uniform_(self.net[-1].weight, -0.0001, 0.0001)
        nn.init.uniform_(self.net[-1].bias, -0.0001, 0.0001)

    def forward(self, coords):
        if self.use_fourier:
            features = self.encoding(coords)
        else:
            features = coords
        raw_coeffs = self.net(features)
        if self.output_raw:
            return raw_coeffs
        coeffs = self.a_max * torch.tanh(raw_coeffs)
        return coeffs