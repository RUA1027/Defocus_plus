# 这个文件中的网络是用来确定 PSF 卷积核 K 的参数（Zernike 像差系数）
# 它们根据输入的图像坐标，预测出对应位置的像差系数
# 这些像差系数描述了光学系统在该位置的失真情况，从而帮助后续的图像复原过程
'''
目标：给定图像平面上的空间位置 (u,v)
   ↓
   输出该位置处的光学像差参数（15个Zernike系数）
   ↓
   用于生成该位置的点扩散函数(PSF)
'''
import torch
import torch.nn as nn
import numpy as np

class FourierFeatureEncoding(nn.Module):
    # 将 2 维的坐标映射到了高维空间（默认 128 维），通过正弦和余弦函数的组合，它能将微小的坐标变化放大
    # 从而让网络能够学习到复杂的、局部化的空间像差分布（例如因镜头组对准不良导致的局部畸变）
    # 没有该类时：直接把坐标（x, y）送入网络，如果图像边缘的像差和中心差异很大（高频变化），网络很难捕捉到这种精细的空间差异，预测出的像差图会过度平缓。
    # 有了它，网络就能区分图像中极小范围内的像差波动，从而实现更精准的空间变化去卷积。
    def __init__(self, input_dim=2, mapping_size=64, scale=10):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.scale = scale
        # scale 越大，生成的特征频率越高，网络能捕捉越精细的细节;越小，特征越平滑。
        # Random Gaussian matrix for mapping，B矩阵决定了“频率”和“方向”
       
        self.register_buffer('B', torch.randn((input_dim, mapping_size)) * scale)

    def forward(self, x):
        # x: [Batch, input_dim]
        # x @ B: [Batch, mapping_size]
        # 2 * pi * x @ B
        xp = 2 * np.pi * x @ self.B
        # 非线性映射：对投影后的结果分别取正弦和余弦
        return torch.cat([torch.sin(xp), torch.cos(xp)], dim=-1)

class AberrationNet(nn.Module):
    # 基于 MLP 的像差建模器，通过神经网络直接学习坐标到像差系数的映射关系
    # 它的主要任务是根据输入的图像坐标，预测出对应位置的像差系数。
    # 这些像差系数描述了光学系统在该位置的失真情况，从而帮助后续的图像复原过程。
    # 在自监督训练中，由于我们没有真实的像差标签，这个网络会根据“图像修复质量好坏”产生的梯度来调整自己，最终学会整张图像各个区域的退化规律。
    '''
    输入坐标 [B, 2]
    ↓
┌─────────────────────────────────────┐
│   FourierFeatureEncoding (可选)     │
│   [B, 2] → [B, 64]                 │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│            MLP 网络                 │
│  Linear(64, 64)                    │
│  LeakyReLU(0.2)                    │
│  Linear(64, 128)                   │
│  LeakyReLU(0.2)                    │
│  Linear(128, 64)                   │
│  LeakyReLU(0.2)                    │
│  Linear(64, 15)  ← 输出层          │
└─────────────────────────────────────┘
    ↓
  raw_coeffs [B, 15]
    ↓
┌─────────────────────────────────────┐
│   Tanh 约束                         │
│   coeffs = a_max × tanh(raw_coeffs) │
│   范围：[-a_max, +a_max]           │
└─────────────────────────────────────┘
    ↓
输出系数 [B, 15]
    '''
    def __init__(self, num_coeffs, hidden_dim, a_max, use_fourier=True):
        super().__init__()
        self.num_coeffs = num_coeffs
        self.a_max = a_max
        self.use_fourier = use_fourier
        
        if use_fourier:
            self.encoding = FourierFeatureEncoding(input_dim=2, mapping_size=hidden_dim//2, scale=5)
            in_dim = hidden_dim # sin + cos -> 2 * mapping_size
        else:
            in_dim = 2

        # 定义了一个包含 4 个线性层（Linear）和 LeakyReLU 激活函数的序列模型。
        # 它负责从高维特征中学习复杂的逻辑映射。    
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, num_coeffs)
        )
        
        # Initialize output layer to near zero to start with near-ideal optics
        # This helps stability - starting from "no aberration" state.
        nn.init.uniform_(self.net[-1].weight, -1e-4, 1e-4)
        nn.init.uniform_(self.net[-1].bias, -1e-4, 1e-4)

    def forward(self, coords):
        """
        coords: [B, 2] in range [-1, 1]
        Returns: coefficients [B, num_coeffs]
        """
        if self.use_fourier:
            features = self.encoding(coords)
        else:
            features = coords
            
        raw_coeffs = self.net(features)
        
        # Constrain coefficients
        coeffs = self.a_max * torch.tanh(raw_coeffs)
        
        return coeffs
